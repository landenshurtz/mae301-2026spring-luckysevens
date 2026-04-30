from __future__ import annotations

import contextlib
import importlib
import importlib.machinery
import importlib.util
import io
import json
import math
import os
import socket
import sys
import tempfile
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

# This order mirrors TxtRunLogger.FIELDNAMES in proxy_fixed.py. The live proxy
# stream sends these fields as JSON after the proxy has made its forward/drop/
# delay decision.
PROXY_LOG_FIELDNAMES = [
    "run_id",
    "packet_index",
    "time_unix_s",
    "t_rel_s",
    "iso_time_utc",
    "direction",
    "packet_len_bytes",
    "mavlink_msg_types",
    "pos_valid",
    "position_source",
    "x_m",
    "y_m",
    "z_m",
    "vx_m_s",
    "vy_m_s",
    "vz_m_s",
    "speed_m_s",
    "lat_deg",
    "lon_deg",
    "rel_alt_m",
    "gps_fix_type",
    "satellites_visible",
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
    "rssi",
    "remrssi",
    "noise",
    "remnoise",
    "rxerrors",
    "fixed_errors",
    "distance_to_hotspot_m",
    "hotspot_strength",
    "center_x_m",
    "center_y_m",
    "radius_m",
    "blackout_radius_m",
    "falloff_power",
    "max_random_loss_prob",
    "max_base_latency_s",
    "max_jitter_s",
    "max_keep_every_n",
    "max_burst_trigger_prob",
    "max_burst_duration_s",
    "burst_active",
    "throttle_counter",
    "queue_depth_before",
    "decision",
    "delivered_label",
    "drop_reason",
    "scheduled_delay_s",
    "total_in",
    "total_forwarded",
    "total_dropped",
    "total_returned",
]

DEFAULT_EXPECTED_COLUMN_COUNT = 22
DEFAULT_INPUT_WINDOW = 3
DEFAULT_FORECAST_HORIZON = 20
DEFAULT_LABEL_INDICES = (20, 21)
DEFAULT_POLYNOMIAL_DEGREE = 1


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        number = float(text)
    except ValueError:
        return default
    if not math.isfinite(number):
        return default
    return number


def _as_int(value: Any, default: int = 0) -> int:
    return int(round(_as_float(value, float(default))))


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _logistic(value: float) -> float:
    if value >= 50.0:
        return 1.0
    if value <= -50.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def _safe_csv_field(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    # dataRefiner.refine_file uses a simple comma split, so avoid commas and
    # newlines rather than relying on CSV quoting.
    return text.replace(",", "|").replace("\r", " ").replace("\n", " ")


def _find_local_module(module_name: str) -> Tuple[Optional[Any], str]:
    """Load a source module or same-folder .pyc module without requiring __pycache__."""
    try:
        return importlib.import_module(module_name), "imported normally"
    except Exception as normal_import_error:
        base_dir = Path(__file__).resolve().parent
        version_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
        candidates = [
            base_dir / f"{module_name}.py",
            base_dir / f"{module_name}.pyc",
            base_dir / f"{module_name}.{version_tag}.pyc",
            base_dir / "__pycache__" / f"{module_name}.{version_tag}.pyc",
            # The uploaded files are CPython 3.11 bytecode named this way.
            base_dir / f"{module_name}.cpython-311.pyc",
            base_dir / "__pycache__" / f"{module_name}.cpython-311.pyc",
        ]
        errors: List[str] = [f"normal import failed: {normal_import_error}"]
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                if candidate.suffix == ".py":
                    spec = importlib.util.spec_from_file_location(module_name, str(candidate))
                    if spec is None or spec.loader is None:
                        errors.append(f"{candidate.name}: no import spec")
                        continue
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    return module, f"loaded {candidate.name}"
                loader = importlib.machinery.SourcelessFileLoader(module_name, str(candidate))
                spec = importlib.util.spec_from_loader(module_name, loader)
                if spec is None:
                    errors.append(f"{candidate.name}: no sourceless spec")
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                loader.exec_module(module)
                return module, f"loaded {candidate.name}"
            except Exception as exc:
                errors.append(f"{candidate.name}: {exc}")
        return None, "; ".join(errors)


class LiveRiskPipeline:
    """
    Turns proxy decision rows into live risk scores.

    The preferred path is:
        proxy JSON row -> dataRefiner.refine_file/refiner adapter -> ai_model.predict

    If the compiled .pyc files cannot be loaded on this Python version, the
    class still prints a deterministic fallback score so mission-control output
    remains live instead of failing the flight script.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        print_period_s: float = 1.0,
        train_online: bool = True,
    ) -> None:
        self.print_period_s = max(0.1, float(print_period_s))
        self.train_online = train_online
        self.last_print_time = 0.0
        self.last_file_refiner_output: Optional[str] = None
        self.last_printed_packet_index: Optional[int] = None
        self.raw_lines: Deque[str] = deque(maxlen=256)
        self.refined_rows: Deque[List[float]] = deque(maxlen=512)
        self.recent_events: Deque[Dict[str, Any]] = deque(maxlen=80)
        self.raw_52_window: Deque[Tuple[float, float]] = deque(maxlen=256)
        self.last_default_quarter: Optional[Tuple[int, int]] = None
        self.warnings_seen: set[str] = set()
        self.score_count = 0

        self.data_refiner, self.data_refiner_status = _find_local_module("dataRefiner")
        self.ai_model, self.ai_model_status = _find_local_module("ai_model")

        self.expected_column_count = int(
            getattr(self.ai_model, "EXPECTED_COLUMN_COUNT", DEFAULT_EXPECTED_COLUMN_COUNT)
            if self.ai_model is not None else DEFAULT_EXPECTED_COLUMN_COUNT
        )
        self.input_window = int(
            getattr(self.ai_model, "INPUT_WINDOW", DEFAULT_INPUT_WINDOW)
            if self.ai_model is not None else DEFAULT_INPUT_WINDOW
        )
        self.forecast_horizon = int(
            getattr(self.ai_model, "FORECAST_HORIZON", DEFAULT_FORECAST_HORIZON)
            if self.ai_model is not None else DEFAULT_FORECAST_HORIZON
        )
        self.label_indices = tuple(
            getattr(self.ai_model, "LABEL_INDICES", DEFAULT_LABEL_INDICES)
            if self.ai_model is not None else DEFAULT_LABEL_INDICES
        )

        self.polynomial_degree = int(os.environ.get("RISK_POLYNOMIAL_DEGREE", DEFAULT_POLYNOMIAL_DEGREE))
        self.weights: Optional[List[List[float]]] = None
        self.model_status = "no model loaded"
        self.last_prediction: Optional[List[float]] = None
        self.model_path = model_path or os.environ.get("RISK_MODEL_PATH", "risk_model.json")
        self._load_model_file()

    def status_lines(self) -> List[str]:
        return [
            f"dataRefiner: {self.data_refiner_status}",
            f"ai_model: {self.ai_model_status}",
            f"model: {self.model_status}",
            f"live rows: expected_columns={self.expected_column_count}, input_window={self.input_window}",
        ]

    def _warn_once(self, key: str, message: str) -> None:
        if key in self.warnings_seen:
            return
        self.warnings_seen.add(key)
        print(f"[RISK WARN] {message}")

    def _load_model_file(self) -> None:
        path = Path(self.model_path)
        if not path.exists():
            self.model_status = (
                f"no {path.name}; using fallback score until online ai_model fit has enough rows"
            )
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                weights = payload.get("weights") or payload.get("model") or payload.get("coef")
                degree = payload.get("polynomial_degree") or payload.get("degree")
                if degree is not None:
                    self.polynomial_degree = int(degree)
            else:
                weights = payload
            normalized = self._normalize_weights(weights)
            if not normalized:
                raise ValueError("model JSON did not contain usable weights")
            self.weights = normalized
            self.model_status = f"loaded {path.name}"
        except Exception as exc:
            self.weights = None
            self.model_status = f"could not load {path.name}: {exc}"

    @staticmethod
    def _normalize_weights(weights: Any) -> Optional[List[List[float]]]:
        if weights is None:
            return None
        if not isinstance(weights, list) or not weights:
            return None
        if all(isinstance(x, (int, float)) for x in weights):
            return [[float(x) for x in weights]]
        normalized: List[List[float]] = []
        for row in weights:
            if not isinstance(row, list) or not row:
                return None
            normalized.append([_as_float(x) for x in row])
        return normalized

    def _proxy_row_to_raw_line(self, row: Dict[str, Any]) -> str:
        payload = dict(row)
        if not payload.get("iso_time_utc"):
            payload["iso_time_utc"] = datetime.now(timezone.utc).isoformat()
        if not payload.get("time_unix_s"):
            payload["time_unix_s"] = f"{time.time():.6f}"
        values = [_safe_csv_field(payload.get(name, "")) for name in PROXY_LOG_FIELDNAMES]
        # The supplied dataRefiner expects 57 comma-separated fields. The proxy
        # logger currently has 55, so append two numeric pads. Field index 52
        # remains total_forwarded, which is numeric and safe for the refiner's
        # rolling average calculation.
        values.extend(["0", "0"])
        return ",".join(values)

    def _run_data_refiner(self, row: Dict[str, Any]) -> Optional[List[float]]:
        if self.data_refiner is None or not hasattr(self.data_refiner, "refine_file"):
            return None
        raw_line = self._proxy_row_to_raw_line(row)
        self.raw_lines.append(raw_line)
        stem = f"ardupilot_live_risk_{os.getpid()}_{id(self)}"
        input_path = Path(tempfile.gettempdir()) / f"{stem}_raw.txt"
        output_path = Path(tempfile.gettempdir()) / f"{stem}_refined.txt"
        try:
            input_path.write_text("\n".join(self.raw_lines) + "\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.data_refiner.refine_file(str(input_path), str(output_path))
            if not output_path.exists():
                return None
            lines = [line.strip() for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                return None
            latest = lines[-1]
            if latest == self.last_file_refiner_output:
                return None
            self.last_file_refiner_output = latest
            numeric = self._numeric_fields(latest.split(","))
            if len(numeric) >= self.expected_column_count:
                return numeric[: self.expected_column_count]
            self._warn_once(
                "short_refiner_output",
                f"dataRefiner output had {len(numeric)} numeric columns; using live refiner fallback for model input",
            )
            return None
        except Exception as exc:
            self._warn_once("data_refiner_failed", f"dataRefiner live pass failed: {exc}; using fallback refiner")
            return None

    @staticmethod
    def _numeric_fields(fields: Iterable[Any]) -> List[float]:
        values: List[float] = []
        for field in fields:
            text = str(field).strip()
            if not text:
                continue
            try:
                value = float(text)
            except ValueError:
                continue
            if math.isfinite(value):
                values.append(value)
        return values

    def _default_refine_row(self, row: Dict[str, Any]) -> Optional[List[float]]:
        """Quarter-second live refiner that emits the 22 numeric columns ai_model expects."""
        event_time = _as_float(row.get("time_unix_s"), time.time())
        quarter = (int(event_time), int((event_time % 1.0) / 0.25))
        if quarter == self.last_default_quarter:
            return None
        self.last_default_quarter = quarter

        decision = str(row.get("decision", "")).lower()
        delivered = _as_float(row.get("delivered_label"), 1.0 if decision != "drop" else 0.0)
        dropped = 1.0 if decision == "drop" or delivered <= 0.0 else 0.0
        scheduled_delay = _as_float(row.get("scheduled_delay_s"), 0.0)
        strength = _as_float(row.get("hotspot_strength"), 0.0)
        distance = _as_float(row.get("distance_to_hotspot_m"), 0.0)
        fix_type = _as_float(row.get("gps_fix_type"), 3.0)
        sats = _as_float(row.get("satellites_visible"), 10.0)
        gps_quality = _clamp01((fix_type / 6.0) * min(1.0, sats / 12.0))

        features = [
            _as_float(row.get("pos_valid")),
            _as_float(row.get("x_m")),
            _as_float(row.get("y_m")),
            _as_float(row.get("z_m")),
            _as_float(row.get("vx_m_s")),
            _as_float(row.get("vy_m_s")),
            _as_float(row.get("vz_m_s")),
            _as_float(row.get("speed_m_s")),
            _as_float(row.get("rel_alt_m")),
            _as_float(row.get("roll_rad")),
            _as_float(row.get("pitch_rad")),
            _as_float(row.get("yaw_rad")),
            distance,
            strength,
            _as_float(row.get("max_random_loss_prob")),
            _as_float(row.get("max_base_latency_s")),
            _as_float(row.get("max_jitter_s")),
            _as_float(row.get("queue_depth_before")),
            scheduled_delay,
            gps_quality,
            dropped,
            delivered,
        ]
        if len(features) < self.expected_column_count:
            features.extend([0.0] * (self.expected_column_count - len(features)))
        return features[: self.expected_column_count]

    def _record_recent_event(self, row: Dict[str, Any]) -> None:
        decision = str(row.get("decision", "")).lower()
        delivered = _as_float(row.get("delivered_label"), 1.0 if decision != "drop" else 0.0)
        self.recent_events.append(
            {
                "decision": decision,
                "dropped": 1.0 if decision == "drop" or delivered <= 0.0 else 0.0,
                "delay": _as_float(row.get("scheduled_delay_s")),
                "strength": _as_float(row.get("hotspot_strength")),
                "queue": _as_float(row.get("queue_depth_before")),
                "fix_type": _as_float(row.get("gps_fix_type"), 3.0),
                "sats": _as_float(row.get("satellites_visible"), 10.0),
            }
        )

    def _fallback_risk_score(self, row: Dict[str, Any]) -> float:
        if not self.recent_events:
            return 0.0
        drops = sum(event["dropped"] for event in self.recent_events) / len(self.recent_events)
        strength = _as_float(row.get("hotspot_strength"))
        scheduled_delay = _as_float(row.get("scheduled_delay_s"))
        base_latency = _as_float(row.get("max_base_latency_s"))
        jitter = _as_float(row.get("max_jitter_s"))
        delay_limit = max(base_latency + jitter, 0.001)
        delay_risk = _clamp01(scheduled_delay / delay_limit)
        queue_risk = _clamp01(_as_float(row.get("queue_depth_before")) / 10.0)
        fix_type = _as_float(row.get("gps_fix_type"), 3.0)
        sats = _as_float(row.get("satellites_visible"), 10.0)
        gps_risk = max(0.0, 1.0 - min(1.0, fix_type / 3.0) * min(1.0, sats / 8.0))
        current_drop = 1.0 if str(row.get("decision", "")).lower() == "drop" else 0.0
        score = max(
            current_drop,
            0.65 * strength + 0.35 * drops,
            0.70 * delay_risk + 0.30 * queue_risk,
            0.45 * gps_risk,
        )
        return _clamp01(score)

    def _maybe_fit_online_model(self) -> None:
        if not self.train_online or self.ai_model is None:
            return
        if not hasattr(self.ai_model, "make_instances") or not hasattr(self.ai_model, "fit_linear_regression"):
            return
        # Keep startup light: fit after enough quarter-second rows, then refresh occasionally.
        if len(self.refined_rows) < max(30, self.input_window + self.forecast_horizon + 2):
            return
        if self.score_count % 20 != 0 and self.weights is not None:
            return
        try:
            rows = list(self.refined_rows)
            instances = self.ai_model.make_instances(
                rows,
                self.input_window,
                min(self.forecast_horizon, max(1, len(rows) - self.input_window - 1)),
                self.label_indices,
                False,
            )
            if len(instances) < 8:
                return
            X = [features for features, _output in instances]
            y = [output for _features, output in instances]
            weights, _std_errors = self.ai_model.fit_linear_regression(X, y, 0.01)
            normalized = self._normalize_weights(weights)
            if normalized:
                self.weights = normalized
                self.model_status = f"online ai_model fit using {len(instances)} live instances"
        except Exception as exc:
            self._warn_once("online_fit_failed", f"online ai_model fit failed: {exc}")

    def _predict_with_model(self, features: List[float]) -> Tuple[Optional[float], Optional[List[float]], str]:
        if self.ai_model is None or self.weights is None or not hasattr(self.ai_model, "predict"):
            return None, None, "fallback"
        # Try the configured degree first, then a few common degrees if the
        # loaded model was saved without metadata.
        degrees = [self.polynomial_degree]
        for degree in (1, 2, 3):
            if degree not in degrees:
                degrees.append(degree)
        last_error: Optional[Exception] = None
        for degree in degrees:
            try:
                prediction = self.ai_model.predict(self.weights, features, degree)
                pred_values = [float(value) for value in prediction]
                self.polynomial_degree = degree
                self.last_prediction = pred_values
                raw = pred_values[0] if pred_values else 0.0
                # If the model is already producing a probability/label-like
                # value, clamp it. Otherwise map regression output to 0..1.
                if -0.25 <= raw <= 1.25:
                    score = _clamp01(raw)
                else:
                    score = _logistic(raw)
                return score, pred_values, "ai_model"
            except Exception as exc:
                last_error = exc
                continue
        if last_error is not None:
            self._warn_once("predict_failed", f"ai_model.predict failed: {last_error}; using fallback score")
        return None, None, "fallback"

    def process_proxy_row(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._record_recent_event(row)

        refined = self._run_data_refiner(row)
        if refined is None:
            refined = self._default_refine_row(row)
        if refined is None:
            return None

        self.refined_rows.append(refined)
        self.score_count += 1
        self._maybe_fit_online_model()

        fallback_score = self._fallback_risk_score(row)
        model_score: Optional[float] = None
        prediction: Optional[List[float]] = None
        source = "fallback"

        if len(self.refined_rows) >= self.input_window:
            model_features: List[float] = []
            for refined_row in list(self.refined_rows)[-self.input_window:]:
                model_features.extend(refined_row)
            model_score, prediction, source = self._predict_with_model(model_features)

        score = fallback_score if model_score is None else model_score
        packet_index = _as_int(row.get("packet_index"), -1)
        now = time.monotonic()
        if now - self.last_print_time < self.print_period_s and packet_index == self.last_printed_packet_index:
            return None
        if now - self.last_print_time < self.print_period_s:
            return None
        self.last_print_time = now
        self.last_printed_packet_index = packet_index

        return {
            "score": _clamp01(score),
            "source": source,
            "fallback_score": fallback_score,
            "prediction": prediction,
            "packet_index": packet_index,
            "decision": row.get("decision", ""),
            "drop_reason": row.get("drop_reason", ""),
            "hotspot_strength": _as_float(row.get("hotspot_strength")),
            "distance_to_hotspot_m": _as_float(row.get("distance_to_hotspot_m")),
            "scheduled_delay_s": _as_float(row.get("scheduled_delay_s")),
            "queue_depth_before": _as_float(row.get("queue_depth_before")),
            "total_in": _as_int(row.get("total_in")),
            "total_dropped": _as_int(row.get("total_dropped")),
            "model_status": self.model_status,
        }


class ProxyRiskListener:
    def __init__(
        self,
        listen_host: str = "0.0.0.0",
        listen_port: int = 14600,
        print_period_s: float = 1.0,
        model_path: Optional[str] = None,
    ) -> None:
        self.listen_addr = (listen_host, int(listen_port))
        train_online = os.environ.get("RISK_TRAIN_ONLINE", "1").strip().lower() not in {"0", "false", "no"}
        self.pipeline = LiveRiskPipeline(
            model_path=model_path,
            print_period_s=print_period_s,
            train_online=train_online,
        )
        self.sock: Optional[socket.socket] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.received = 0

    def start(self) -> None:
        if self.running:
            return
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.listen_addr)
        self.sock.settimeout(0.25)
        self.running = True
        self.thread = threading.Thread(target=self._loop, name="proxy-risk-listener", daemon=True)
        self.thread.start()
        print(f"[RISK] Listening for proxy risk stream on UDP {self.listen_addr[0]}:{self.listen_addr[1]}")
        for line in self.pipeline.status_lines():
            print(f"[RISK] {line}")

    def stop(self) -> None:
        self.running = False
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def _loop(self) -> None:
        while self.running:
            try:
                assert self.sock is not None
                packet, _src = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                row = json.loads(packet.decode("utf-8"))
                if not isinstance(row, dict):
                    continue
                self.received += 1
                result = self.pipeline.process_proxy_row(row)
                if result is not None:
                    self._print_result(result)
            except Exception as exc:
                print(f"[RISK ERROR] {exc}")

    @staticmethod
    def _print_result(result: Dict[str, Any]) -> None:
        prediction = result.get("prediction")
        if prediction is None:
            pred_text = ""
        else:
            shown = ",".join(f"{value:.3f}" for value in prediction[:3])
            pred_text = f" pred=[{shown}]"
        reason = str(result.get("drop_reason") or "")
        reason_text = f" reason={reason}" if reason else ""
        print(
            "[RISK] "
            f"score={result['score']:.3f} "
            f"source={result['source']} "
            f"packet={result['packet_index']} "
            f"decision={result.get('decision', '')}{reason_text} "
            f"strength={result['hotspot_strength']:.3f} "
            f"distance={result['distance_to_hotspot_m']:.1f}m "
            f"delay={result['scheduled_delay_s']:.3f}s "
            f"queue={result['queue_depth_before']:.0f} "
            f"drops={result['total_dropped']}/{result['total_in']}"
            f"{pred_text}"
        )
