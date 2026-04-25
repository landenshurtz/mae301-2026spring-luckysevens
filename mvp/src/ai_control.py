# Adding AI Backend
import socket
import threading
import time
from collections import deque
from typing import Optional

from fastapi import FastAPI
import uvicorn
from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavlink2


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = {
            "telemetry": {},
            "features": {},
            "decision": {},
        }

    def update(self, telemetry, features, decision):
        with self.lock:
            self.status = {
                "telemetry": telemetry,
                "features": features,
                "decision": decision,
            }

    def get(self):
        with self.lock:
            return self.status.copy()


class MissionAIController:
    """
    Receives impaired telemetry from Joey's proxy on UDP 14550.
    Sends MAVLink command packets back to Joey's proxy on UDP 14551.

    Joey's existing proxy architecture:
      SITL/MAVProxy -> proxy(14560) -> Mission Control(14550)
      Mission Control replies -> proxy(14551) -> SITL/MAVProxy
    """

    def __init__(
        self,
        listen_host: str = "127.0.0.1",
        listen_port: int = 14550,
        reply_host: str = "127.0.0.1",
        reply_port: int = 14551,
        api_host: str = "127.0.0.1",
        api_port: int = 8000,
        enable_vehicle_commands: bool = False,
    ):
        self.listen_addr = (listen_host, listen_port)
        self.reply_addr = (reply_host, reply_port)
        self.api_host = api_host
        self.api_port = api_port
        self.enable_vehicle_commands = enable_vehicle_commands

        self.sock_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_in.bind(self.listen_addr)
        self.sock_in.settimeout(0.1)

        self.sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.mav_parser = mavlink2.MAVLink(None)

        # Outgoing MAVLink encoder
        self.mav_out_file = mavutil.mavfile(None)
        self.mav_sender = mavlink2.MAVLink(self.mav_out_file)
        self.mav_sender.srcSystem = 250
        self.mav_sender.srcComponent = 1

        self.running = False
        self.shared_state = SharedState()

        # Telemetry memory
        self.last_packet_time: Optional[float] = None
        self.inter_arrival_ms = deque(maxlen=50)

        self.last_heartbeat_time: Optional[float] = None
        self.heartbeat_interval_ms: float = 1000.0
        self.missed_heartbeats: int = 0
        self.dropout_start_time: Optional[float] = None
        self.dropout_streak_s: float = 0.0

        self.packet_counter_window = deque(maxlen=50)
        self.forward_timestamps = deque(maxlen=50)

        # Current vehicle state
        self.mode = "UNKNOWN"
        self.armed = False
        self.altitude_m = None
        self.groundspeed_mps = None
        self.lat = None
        self.lon = None
        self.heading_deg = None
        self.battery_pct = None
        self.gps_hdop = None
        self.waypoint_index = None
        self.local_x = None
        self.local_y = None
        self.local_z = None

        # Decision state
        self.last_action = "CONTINUE"
        self.last_action_sent = None
        self.last_action_send_time = 0.0
        self.link_stable_since = time.time()

        # For receiving target system/component from live traffic
        self.target_system = 1
        self.target_component = 1

        # Loop timing
        self.poll_hz = 10.0
        self.action_hold_s = 3.0
        self.stable_required_s = 5.0

    # -----------------------------
    # Telemetry parsing
    # -----------------------------
    def parse_packet(self, packet: bytes):
        now = time.time()

        if self.last_packet_time is not None:
            dt_ms = (now - self.last_packet_time) * 1000.0
            self.inter_arrival_ms.append(dt_ms)
        self.last_packet_time = now
        self.packet_counter_window.append(now)

        parsed_any = False

        for byte_value in packet:
            try:
                msg = self.mav_parser.parse_char(bytes([byte_value]))
                if msg is None:
                    continue

                parsed_any = True

                if hasattr(msg, "get_srcSystem"):
                    try:
                        self.target_system = msg.get_srcSystem()
                    except Exception:
                        pass

                if hasattr(msg, "get_srcComponent"):
                    try:
                        self.target_component = msg.get_srcComponent()
                    except Exception:
                        pass

                msg_type = msg.get_type()

                if msg_type == "HEARTBEAT":
                    self.handle_heartbeat(msg, now)

                elif msg_type == "GLOBAL_POSITION_INT":
                    if hasattr(msg, "relative_alt"):
                        self.altitude_m = float(msg.relative_alt) / 1000.0
                    if hasattr(msg, "lat"):
                        self.lat = float(msg.lat) / 1e7
                    if hasattr(msg, "lon"):
                        self.lon = float(msg.lon) / 1e7
                    if hasattr(msg, "hdg") and msg.hdg != 65535:
                        self.heading_deg = float(msg.hdg) / 100.0

                elif msg_type == "VFR_HUD":
                    if hasattr(msg, "groundspeed"):
                        self.groundspeed_mps = float(msg.groundspeed)

                elif msg_type == "SYS_STATUS":
                    if hasattr(msg, "battery_remaining"):
                        self.battery_pct = float(msg.battery_remaining)

                elif msg_type == "GPS_RAW_INT":
                    if hasattr(msg, "eph") and msg.eph != 65535:
                        self.gps_hdop = float(msg.eph) / 100.0

                elif msg_type == "MISSION_CURRENT":
                    if hasattr(msg, "seq"):
                        self.waypoint_index = int(msg.seq)

                elif msg_type == "LOCAL_POSITION_NED":
                    if hasattr(msg, "x"):
                        self.local_x = float(msg.x)
                    if hasattr(msg, "y"):
                        self.local_y = float(msg.y)
                    if hasattr(msg, "z"):
                        self.local_z = float(msg.z)

            except Exception:
                pass

        if not parsed_any:
            self.handle_possible_dropout(now)
        else:
            self.update_dropout(now)

    def handle_heartbeat(self, msg, now: float):
        try:
            self.mode = mavutil.mode_string_v10(msg)
        except Exception:
            pass

        try:
            base_mode = getattr(msg, "base_mode", 0)
            self.armed = bool(base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        except Exception:
            pass

        if self.last_heartbeat_time is not None:
            interval_ms = (now - self.last_heartbeat_time) * 1000.0
            self.heartbeat_interval_ms = interval_ms
            if interval_ms > 1500.0:
                self.missed_heartbeats += max(0, int(interval_ms // 1000.0) - 1)
            else:
                self.missed_heartbeats = max(0, self.missed_heartbeats - 1)

        self.last_heartbeat_time = now

    def handle_possible_dropout(self, now: float):
        if self.last_packet_time is None:
            return
        gap_s = now - self.last_packet_time
        if gap_s > 0.5:
            if self.dropout_start_time is None:
                self.dropout_start_time = self.last_packet_time
            self.dropout_streak_s = now - self.dropout_start_time

    def update_dropout(self, now: float):
        if self.last_packet_time is None:
            return
        gap_s = now - self.last_packet_time
        if gap_s > 0.5:
            if self.dropout_start_time is None:
                self.dropout_start_time = self.last_packet_time
            self.dropout_streak_s = now - self.dropout_start_time
        else:
            self.dropout_start_time = None
            self.dropout_streak_s = 0.0

    # -----------------------------
    # Feature computation
    # -----------------------------
    def compute_features(self):
        now = time.time()

        if len(self.inter_arrival_ms) > 0:
            mean_dt = sum(self.inter_arrival_ms) / len(self.inter_arrival_ms)
            var_dt = sum((x - mean_dt) ** 2 for x in self.inter_arrival_ms) / len(self.inter_arrival_ms)
            jitter_ms = var_dt ** 0.5
            latency_ms = max(0.0, mean_dt)
            delayed = sum(1 for x in self.inter_arrival_ms if x > 300.0)
            packet_loss_pct = 100.0 * delayed / len(self.inter_arrival_ms)
        else:
            latency_ms = 0.0
            jitter_ms = 0.0
            packet_loss_pct = 0.0

        if self.last_packet_time is not None:
            self.update_dropout(now)

        features = {
            "heartbeat_interval_ms": float(self.heartbeat_interval_ms),
            "missed_heartbeats": int(self.missed_heartbeats),
            "packet_loss_pct": float(packet_loss_pct),
            "latency_ms": float(latency_ms),
            "jitter_ms": float(jitter_ms),
            "dropout_streak_s": float(self.dropout_streak_s),
        }
        return features

    @staticmethod
    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def compute_risk(self, features):
        heartbeat_term = self.clamp01((features["heartbeat_interval_ms"] - 1000.0) / 1500.0)
        missed_term = self.clamp01(features["missed_heartbeats"] / 6.0)
        packet_loss_term = self.clamp01(features["packet_loss_pct"] / 100.0)
        latency_term = self.clamp01(features["latency_ms"] / 500.0)
        jitter_term = self.clamp01(features["jitter_ms"] / 200.0)
        dropout_term = self.clamp01(features["dropout_streak_s"] / 5.0)

        score = (
            0.12 * heartbeat_term +
            0.16 * missed_term +
            0.22 * packet_loss_term +
            0.18 * latency_term +
            0.14 * jitter_term +
            0.18 * dropout_term
        )

        severe_count = 0
        if features["packet_loss_pct"] >= 35.0:
            severe_count += 1
        if features["latency_ms"] >= 250.0:
            severe_count += 1
        if features["jitter_ms"] >= 100.0:
            severe_count += 1
        if features["missed_heartbeats"] >= 3:
            severe_count += 1
        if features["dropout_streak_s"] >= 1.0:
            severe_count += 1

        if severe_count >= 3:
            score += 0.15
        elif severe_count >= 2:
            score += 0.08

        score = self.clamp01(score)

        if score < 0.30:
            label = "LOW"
        elif score < 0.65:
            label = "MEDIUM"
        elif score < 0.85:
            label = "HIGH"
        else:
            label = "CRITICAL"

        return score, label

    def choose_action(self, risk_score: float, features):
        now = time.time()

        stable = (
            risk_score < 0.30 and
            features["packet_loss_pct"] < 10.0 and
            features["latency_ms"] < 120.0 and
            features["jitter_ms"] < 40.0 and
            features["missed_heartbeats"] == 0 and
            features["dropout_streak_s"] < 0.2
        )

        if stable:
            stable_for_s = now - self.link_stable_since
        else:
            self.link_stable_since = now
            stable_for_s = 0.0

        if risk_score < 0.30:
            if self.last_action in ["LOITER_AND_CLIMB", "RTL", "SLOW_DOWN"]:
                if stable_for_s >= self.stable_required_s:
                    action = "RESUME_AUTO"
                else:
                    action = self.last_action
            else:
                action = "CONTINUE"

        elif risk_score < 0.65:
            action = "MONITOR"

        elif risk_score < 0.85:
            if features["dropout_streak_s"] >= 1.0 or features["missed_heartbeats"] >= 3:
                action = "LOITER_AND_CLIMB"
            else:
                action = "SLOW_DOWN"

        else:
            if features["dropout_streak_s"] >= 2.0 or features["packet_loss_pct"] >= 60.0:
                action = "RTL"
            else:
                action = "LOITER_AND_CLIMB"

        self.last_action = action
        return action, stable_for_s

    def build_explanation(self, risk_score: float, action: str, features):
        if risk_score < 0.30:
            level_text = "Link quality is stable."
        elif risk_score < 0.65:
            level_text = "Link quality is beginning to degrade."
        elif risk_score < 0.85:
            level_text = "Link quality is degrading and communication loss is becoming more likely."
        else:
            level_text = "Link quality is critical and communication loss is highly likely."

        reasons = []
        if features["packet_loss_pct"] >= 20.0:
            reasons.append("elevated packet loss")
        if features["latency_ms"] >= 250.0:
            reasons.append("high latency")
        if features["jitter_ms"] >= 100.0:
            reasons.append("high jitter")
        if features["missed_heartbeats"] >= 3:
            reasons.append("missed heartbeats")
        if features["dropout_streak_s"] >= 1.0:
            reasons.append("a sustained dropout streak")

        if reasons:
            reason_text = "The main indicators are " + ", ".join(reasons) + "."
        else:
            reason_text = "No major communication anomalies were detected."

        action_text = {
            "CONTINUE": "Continue the mission in AUTO mode.",
            "MONITOR": "Continue the mission but monitor the link closely.",
            "SLOW_DOWN": "Reduce forward speed to lower communication risk.",
            "LOITER_AND_CLIMB": "Switch to GUIDED mode, loiter, and climb to improve line-of-sight.",
            "RTL": "Initiate return-to-launch to protect the vehicle.",
            "RESUME_AUTO": "The link has remained stable long enough to resume AUTO mode.",
        }.get(action, "Maintain safe operation and reassess the link.")

        return f"{level_text} {reason_text} Recommended action: {action_text}"

    # -----------------------------
    # MAVLink command output
    # -----------------------------
    def send_mode_change(self, mode_name: str):
        # ArduPilot common custom modes
        mode_map = {
            "AUTO": 3,
            "GUIDED": 4,
            "LOITER": 5,
            "RTL": 6,
        }

        if mode_name not in mode_map:
            return False

        msg = self.mav_sender.command_long_encode(
            self.target_system,
            self.target_component,
            mavlink2.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_map[mode_name],
            0, 0, 0, 0, 0
        )
        packet = msg.pack(self.mav_sender)
        self.sock_out.sendto(packet, self.reply_addr)
        return True

    def send_loiter_and_climb(self):
        # Simplified MVP behavior:
        # 1) set GUIDED
        # 2) send a reposition target if we have coordinates
        ok = self.send_mode_change("GUIDED")

        if self.lat is not None and self.lon is not None and self.altitude_m is not None:
            target_alt = self.altitude_m + 10.0

            lat_int = int(self.lat * 1e7)
            lon_int = int(self.lon * 1e7)

            type_mask = (
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
            )

            msg = self.mav_sender.set_position_target_global_int_encode(
                0,
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                type_mask,
                lat_int,
                lon_int,
                target_alt,
                0, 0, 0,
                0, 0, 0,
                0, 0
            )
            packet = msg.pack(self.mav_sender)
            self.sock_out.sendto(packet, self.reply_addr)

        return ok

    def execute_action(self, action: str):
        if action in ["CONTINUE", "MONITOR", "SLOW_DOWN"]:
            return False
        if action == "LOITER_AND_CLIMB":
            return self.send_loiter_and_climb()
        if action == "RTL":
            return self.send_mode_change("RTL")
        if action == "RESUME_AUTO":
            return self.send_mode_change("AUTO")
        return False

    # -----------------------------
    # Threads
    # -----------------------------
    def receive_loop(self):
        while self.running:
            try:
                packet, _ = self.sock_in.recvfrom(65535)
                self.parse_packet(packet)
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                self.running = False
            except Exception as exc:
                print(f"[ERROR] receive_loop: {exc}")

    def decision_loop(self):
        while self.running:
            try:
                features = self.compute_features()
                risk_score, risk_label = self.compute_risk(features)
                action, stable_for_s = self.choose_action(risk_score, features)
                explanation = self.build_explanation(risk_score, action, features)

                telemetry = {
                    "timestamp_s": time.time(),
                    "mode": self.mode,
                    "armed": self.armed,
                    "battery_pct": self.battery_pct,
                    "groundspeed_mps": self.groundspeed_mps,
                    "altitude_m": self.altitude_m,
                    "lat": self.lat,
                    "lon": self.lon,
                    "heading_deg": self.heading_deg,
                    "gps_hdop": self.gps_hdop,
                    "waypoint_index": self.waypoint_index,
                    "local_x": self.local_x,
                    "local_y": self.local_y,
                    "local_z": self.local_z,
                }

                action_sent = False
                now = time.time()
                if self.enable_vehicle_commands:
                    if action != self.last_action_sent or (now - self.last_action_send_time) > self.action_hold_s:
                        action_sent = self.execute_action(action)
                        if action_sent:
                            self.last_action_sent = action
                            self.last_action_send_time = now

                decision = {
                    "timestamp_s": time.time(),
                    "risk_score": risk_score,
                    "risk_level": risk_label,
                    "recommended_action": action,
                    "explanation": explanation,
                    "mode_before_action": self.mode,
                    "action_sent": action_sent,
                    "stable_for_s": stable_for_s,
                }

                self.shared_state.update(telemetry, features, decision)

                print("=" * 80)
                print(f"MODE: {self.mode} | WP: {self.waypoint_index} | BAT: {self.battery_pct}")
                print(
                    f"HB(ms): {features['heartbeat_interval_ms']:.1f} | "
                    f"MISSED: {features['missed_heartbeats']} | "
                    f"LOSS(%): {features['packet_loss_pct']:.1f}"
                )
                print(
                    f"LAT(ms): {features['latency_ms']:.1f} | "
                    f"JITTER(ms): {features['jitter_ms']:.1f} | "
                    f"DROPOUT(s): {features['dropout_streak_s']:.2f}"
                )
                print(f"RISK: {risk_score:.2f} ({risk_label}) | ACTION: {action} | SENT: {action_sent}")
                print(f"EXPLANATION: {explanation}")

                time.sleep(1.0 / self.poll_hz)

            except KeyboardInterrupt:
                self.running = False
            except Exception as exc:
                print(f"[ERROR] decision_loop: {exc}")
                time.sleep(0.5)

    def api_loop(self):
        app = FastAPI(title="Mission AI Control", version="1.0")

        @app.get("/health")
        def health():
            return {"status": "ok"}

        @app.get("/status")
        def status():
            return self.shared_state.get()

        uvicorn.run(app, host=self.api_host, port=self.api_port)

    def start(self):
        self.running = True

        print(f"[INFO] Mission AI listening on UDP {self.listen_addr[0]}:{self.listen_addr[1]}")
        print(f"[INFO] Mission AI sending replies to UDP {self.reply_addr[0]}:{self.reply_addr[1]}")
        print(f"[INFO] API status endpoint at http://{self.api_host}:{self.api_port}/status")
        print(f"[INFO] Vehicle commands enabled: {self.enable_vehicle_commands}")

        threading.Thread(target=self.receive_loop, daemon=True).start()
        threading.Thread(target=self.decision_loop, daemon=True).start()
        threading.Thread(target=self.api_loop, daemon=True).start()

        try:
            while self.running:
                time.sleep(0.25)
        except KeyboardInterrupt:
            self.running = False
        finally:
            self.sock_in.close()
            self.sock_out.close()


if __name__ == "__main__":
    controller = MissionAIController(
        listen_host="127.0.0.1",
        listen_port=14550,
        reply_host="127.0.0.1",
        reply_port=14551,
        api_host="127.0.0.1",
        api_port=8000,
        enable_vehicle_commands=False,  # keep False first
    )
    controller.start()
