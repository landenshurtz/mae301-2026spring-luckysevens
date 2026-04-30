import argparse
import csv
import json
import os
import socket
import threading
import time
import random
import heapq
from datetime import datetime, timezone
from math import cos, hypot, radians

from pymavlink.dialects.v20 import common as mavlink2


class DelayQueue:
    def __init__(self):
        self.heap = []
        self.lock = threading.Lock()

    def push(self, send_time, packet):
        with self.lock:
            heapq.heappush(self.heap, (send_time, packet))

    def pop_ready(self, now):
        ready = []
        with self.lock:
            while self.heap and self.heap[0][0] <= now:
                ready.append(heapq.heappop(self.heap))
        return ready

    def size(self):
        with self.lock:
            return len(self.heap)


class TxtRunLogger:
    """
    Writes one row per incoming SITL -> Mission Control packet.
    The file uses CSV formatting even if the extension is .txt, so it can be
    opened with pandas, Excel, or any text editor.
    """

    FIELDNAMES = [
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

    def __init__(self, path, run_id=None):
        self.path = path
        self.run_id = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
        self.start_monotonic = time.monotonic()
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDNAMES)
        self.writer.writeheader()
        self.file.flush()

    @staticmethod
    def _value(value):
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6f}"
        return value

    def write(self, row):
        now = time.time()
        full_row = {
            "run_id": self.run_id,
            "time_unix_s": f"{now:.6f}",
            "t_rel_s": f"{time.monotonic() - self.start_monotonic:.6f}",
            "iso_time_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        }
        full_row.update(row)
        for name in self.FIELDNAMES:
            full_row.setdefault(name, "")
            full_row[name] = self._value(full_row[name])
        with self.lock:
            self.writer.writerow(full_row)
            self.file.flush()

    def close(self):
        with self.lock:
            self.file.flush()
            self.file.close()


class BidirectionalLocalHotspotProxy:
    """
    MAVProxy/SITL -> proxy(14550) -> Mission Control(14551)
    Mission Control replies -> proxy(14552) -> MAVProxy/SITL source port

    The SITL -> Mission Control direction can be impaired and logged.
    The Mission Control -> SITL direction is passed through unchanged so
    mission uploads and command packets are not broken by the proxy itself.
    """

    def __init__(
        self,
        listen_host="0.0.0.0",
        sitl_listen_port=14550,
        mission_host="127.0.0.1",
        mission_port=14551,
        mission_reply_port=14552,
        log_file="signal_loss_run.txt",
        run_id=None,
        risk_stream_host="127.0.0.1",
        risk_stream_port=14600,
        risk_stream_enabled=True,
    ):
        self.listen_addr = (listen_host, sitl_listen_port)
        self.mission_addr = (mission_host, mission_port)
        self.mission_reply_addr = (listen_host, mission_reply_port)
        self.risk_stream_addr = (risk_stream_host, int(risk_stream_port))
        self.risk_stream_enabled = bool(risk_stream_enabled)

        self.sock_sitl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_sitl.bind(self.listen_addr)
        self.sock_sitl.settimeout(0.1)

        self.sock_mission = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_mission.bind(self.mission_reply_addr)
        self.sock_mission.settimeout(0.1)

        self.sock_risk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.delay_queue = DelayQueue()
        self.mav_parser = mavlink2.MAVLink(None)
        self.logger = TxtRunLogger(log_file, run_id=run_id)

        self.running = False
        self.last_sitl_addr = None

        self.total_in = 0
        self.total_forwarded = 0
        self.total_dropped = 0
        self.total_returned = 0
        self.packet_index = 0

        # Local position tracking from LOCAL_POSITION_NED.
        self.pos_valid = False
        self.x_m = 0.0
        self.y_m = 0.0
        self.z_m = 0.0
        self.vx_m_s = None
        self.vy_m_s = None
        self.vz_m_s = None
        self.speed_m_s = None
        self.lat_deg = None
        self.lon_deg = None
        self.rel_alt_m = None
        self.gps_fix_type = None
        self.satellites_visible = None
        self.roll_rad = None
        self.pitch_rad = None
        self.yaw_rad = None
        self.rssi = None
        self.remrssi = None
        self.noise = None
        self.remnoise = None
        self.rxerrors = None
        self.fixed_errors = None

        # If LOCAL_POSITION_NED is not present, estimate local N/E/D meters
        # from the first GLOBAL_POSITION_INT message. This makes the log useful
        # even when MAVProxy is only streaming GPS/global position messages.
        self.global_origin_lat_deg = None
        self.global_origin_lon_deg = None
        self.position_source = "none"

        # Hotspot configuration.
        self.center_x_m = 50.0
        self.center_y_m = 25.0
        self.radius_m = 100.0
        self.blackout_radius_m = 0.0
        self.falloff_power = 1.5

        # Impairments are OFF by default for validation.
        self.max_random_loss_prob = 0.0
        self.max_base_latency_s = 0.0
        self.max_jitter_s = 0.0
        self.max_keep_every_n = 1
        self.max_burst_trigger_prob = 0.0
        self.max_burst_duration_s = 0.0

        self.burst_end = 0.0
        self.throttle_counter = 0

    def update_position_from_packet(self, packet):
        msg_types = []
        for byte_value in packet:
            try:
                msg = self.mav_parser.parse_char(bytes([byte_value]))
                if msg is None:
                    continue
                msg_type = msg.get_type()
                msg_types.append(msg_type)
                if msg_type == "LOCAL_POSITION_NED":
                    self.x_m = float(msg.x)
                    self.y_m = float(msg.y)
                    self.z_m = float(msg.z)
                    self.vx_m_s = float(getattr(msg, "vx", 0.0))
                    self.vy_m_s = float(getattr(msg, "vy", 0.0))
                    self.vz_m_s = float(getattr(msg, "vz", 0.0))
                    self.speed_m_s = (self.vx_m_s ** 2 + self.vy_m_s ** 2 + self.vz_m_s ** 2) ** 0.5
                    self.position_source = "LOCAL_POSITION_NED"
                    self.pos_valid = True
                elif msg_type == "GLOBAL_POSITION_INT":
                    self.lat_deg = float(msg.lat) / 1e7
                    self.lon_deg = float(msg.lon) / 1e7
                    self.rel_alt_m = float(msg.relative_alt) / 1000.0

                    # GLOBAL_POSITION_INT velocities are in cm/s in NED axes.
                    self.vx_m_s = float(getattr(msg, "vx", 0.0)) / 100.0
                    self.vy_m_s = float(getattr(msg, "vy", 0.0)) / 100.0
                    self.vz_m_s = float(getattr(msg, "vz", 0.0)) / 100.0
                    self.speed_m_s = (self.vx_m_s ** 2 + self.vy_m_s ** 2 + self.vz_m_s ** 2) ** 0.5

                    # Build a local coordinate estimate from GPS if needed.
                    if self.lat_deg != 0.0 and self.lon_deg != 0.0:
                        if self.global_origin_lat_deg is None:
                            self.global_origin_lat_deg = self.lat_deg
                            self.global_origin_lon_deg = self.lon_deg

                        earth_radius_m = 6371000.0
                        dlat_rad = radians(self.lat_deg - self.global_origin_lat_deg)
                        dlon_rad = radians(self.lon_deg - self.global_origin_lon_deg)
                        self.x_m = dlat_rad * earth_radius_m       # north meters
                        self.y_m = dlon_rad * earth_radius_m * cos(radians(self.global_origin_lat_deg))  # east meters
                        self.z_m = -self.rel_alt_m                 # NED down meters
                        self.position_source = "GLOBAL_POSITION_INT_ESTIMATE"
                        self.pos_valid = True
                elif msg_type == "GPS_RAW_INT":
                    self.gps_fix_type = int(getattr(msg, "fix_type", 0))
                    self.satellites_visible = int(getattr(msg, "satellites_visible", 0))
                elif msg_type == "ATTITUDE":
                    self.roll_rad = float(getattr(msg, "roll", 0.0))
                    self.pitch_rad = float(getattr(msg, "pitch", 0.0))
                    self.yaw_rad = float(getattr(msg, "yaw", 0.0))
                elif msg_type == "RADIO_STATUS":
                    self.rssi = int(getattr(msg, "rssi", 0))
                    self.remrssi = int(getattr(msg, "remrssi", 0))
                    self.noise = int(getattr(msg, "noise", 0))
                    self.remnoise = int(getattr(msg, "remnoise", 0))
                    self.rxerrors = int(getattr(msg, "rxerrors", 0))
                    self.fixed_errors = int(getattr(msg, "fixed", 0))
            except Exception:
                pass
        return msg_types

    def distance_to_hotspot(self):
        if not self.pos_valid:
            return None
        return hypot(self.x_m - self.center_x_m, self.y_m - self.center_y_m)

    def compute_strength(self, distance_m):
        if distance_m is None:
            return 0.0
        if distance_m >= self.radius_m:
            return 0.0
        raw_strength = 1.0 - (distance_m / self.radius_m)
        return max(0.0, min(1.0, raw_strength ** self.falloff_power))

    def blackout_active(self, distance_m):
        if distance_m is None:
            return False
        return self.blackout_radius_m > 0.0 and distance_m <= self.blackout_radius_m

    def burst_active(self, now, strength):
        if strength <= 0.0 or self.max_burst_trigger_prob <= 0.0 or self.max_burst_duration_s <= 0.0:
            return False
        if self.burst_end > now:
            return True
        trigger_prob = self.max_burst_trigger_prob * strength
        if random.random() < trigger_prob:
            burst_duration = max(0.01, self.max_burst_duration_s * strength)
            self.burst_end = now + burst_duration
            print(f"[BURST] Started burst for {burst_duration:.3f}s")
            return True
        return False

    def drop_by_random_loss(self, strength):
        if strength <= 0.0 or self.max_random_loss_prob <= 0.0:
            return False
        return random.random() < (self.max_random_loss_prob * strength)

    def drop_by_throttle(self, strength):
        if strength <= 0.0 or self.max_keep_every_n <= 1:
            return False
        keep_every_n = int(round(1 + strength * (self.max_keep_every_n - 1)))
        keep_every_n = max(1, keep_every_n)
        if keep_every_n == 1:
            return False
        self.throttle_counter += 1
        keep = (self.throttle_counter % keep_every_n) == 0
        return not keep

    def compute_delay(self, strength):
        if strength <= 0.0:
            return 0.0
        base_latency = self.max_base_latency_s * strength
        jitter_limit = self.max_jitter_s * strength
        jitter = random.uniform(-jitter_limit, jitter_limit)
        return max(0.0, base_latency + jitter)

    def _base_log_row(self, packet, msg_types, distance_m, strength, queue_depth_before):
        return {
            "packet_index": self.packet_index,
            "direction": "sitl_to_mission",
            "packet_len_bytes": len(packet),
            "mavlink_msg_types": "|".join(msg_types),
            "pos_valid": int(self.pos_valid),
            "position_source": self.position_source,
            "x_m": self.x_m if self.pos_valid else None,
            "y_m": self.y_m if self.pos_valid else None,
            "z_m": self.z_m if self.pos_valid else None,
            "vx_m_s": self.vx_m_s,
            "vy_m_s": self.vy_m_s,
            "vz_m_s": self.vz_m_s,
            "speed_m_s": self.speed_m_s,
            "lat_deg": self.lat_deg,
            "lon_deg": self.lon_deg,
            "rel_alt_m": self.rel_alt_m,
            "gps_fix_type": self.gps_fix_type,
            "satellites_visible": self.satellites_visible,
            "roll_rad": self.roll_rad,
            "pitch_rad": self.pitch_rad,
            "yaw_rad": self.yaw_rad,
            "rssi": self.rssi,
            "remrssi": self.remrssi,
            "noise": self.noise,
            "remnoise": self.remnoise,
            "rxerrors": self.rxerrors,
            "fixed_errors": self.fixed_errors,
            "distance_to_hotspot_m": distance_m,
            "hotspot_strength": strength,
            "center_x_m": self.center_x_m,
            "center_y_m": self.center_y_m,
            "radius_m": self.radius_m,
            "blackout_radius_m": self.blackout_radius_m,
            "falloff_power": self.falloff_power,
            "max_random_loss_prob": self.max_random_loss_prob,
            "max_base_latency_s": self.max_base_latency_s,
            "max_jitter_s": self.max_jitter_s,
            "max_keep_every_n": self.max_keep_every_n,
            "max_burst_trigger_prob": self.max_burst_trigger_prob,
            "max_burst_duration_s": self.max_burst_duration_s,
            "burst_active": int(self.burst_end > time.time()),
            "throttle_counter": self.throttle_counter,
            "queue_depth_before": queue_depth_before,
            "total_in": self.total_in,
            "total_forwarded": self.total_forwarded,
            "total_dropped": self.total_dropped,
            "total_returned": self.total_returned,
        }

    def _emit_risk_row(self, row):
        if not self.risk_stream_enabled:
            return
        now = time.time()
        payload = dict(row)
        payload.setdefault("run_id", self.logger.run_id)
        payload.setdefault("time_unix_s", f"{now:.6f}")
        payload.setdefault("t_rel_s", f"{time.monotonic() - self.logger.start_monotonic:.6f}")
        payload.setdefault("iso_time_utc", datetime.fromtimestamp(now, timezone.utc).isoformat())
        try:
            packet = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
            self.sock_risk.sendto(packet, self.risk_stream_addr)
        except Exception as exc:
            print(f"[WARN] Unable to emit risk stream row: {exc}")

    def _log_decision(self, base_row, decision, delivered_label, drop_reason="", scheduled_delay_s=0.0):
        row = dict(base_row)
        row.update(
            {
                "decision": decision,
                "delivered_label": int(delivered_label),
                "drop_reason": drop_reason,
                "scheduled_delay_s": scheduled_delay_s,
            }
        )
        self.logger.write(row)
        self._emit_risk_row(row)

    def process_sitl_packet(self, packet):
        self.packet_index += 1
        msg_types = self.update_position_from_packet(packet)

        now = time.time()
        distance_m = self.distance_to_hotspot()
        strength = self.compute_strength(distance_m)
        queue_depth_before = self.delay_queue.size()
        base_row = self._base_log_row(packet, msg_types, distance_m, strength, queue_depth_before)

        if self.blackout_active(distance_m):
            self.total_dropped += 1
            self._log_decision(base_row, "drop", 0, drop_reason="blackout", scheduled_delay_s=0.0)
            return

        if self.burst_active(now, strength):
            self.total_dropped += 1
            self._log_decision(base_row, "drop", 0, drop_reason="burst", scheduled_delay_s=0.0)
            return

        if self.drop_by_random_loss(strength):
            self.total_dropped += 1
            self._log_decision(base_row, "drop", 0, drop_reason="random_loss", scheduled_delay_s=0.0)
            return

        if self.drop_by_throttle(strength):
            self.total_dropped += 1
            self._log_decision(base_row, "drop", 0, drop_reason="throttle", scheduled_delay_s=0.0)
            return

        delay = self.compute_delay(strength)
        if delay > 0:
            self.delay_queue.push(now + delay, packet)
            self._log_decision(base_row, "delay", 1, drop_reason="", scheduled_delay_s=delay)
        else:
            self.sock_mission.sendto(packet, self.mission_addr)
            self.total_forwarded += 1
            self._log_decision(base_row, "forward", 1, drop_reason="", scheduled_delay_s=0.0)

    def sitl_loop(self):
        while self.running:
            try:
                packet, src_addr = self.sock_sitl.recvfrom(65535)
                self.last_sitl_addr = src_addr
                self.total_in += 1
                self.process_sitl_packet(packet)
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                self.running = False
            except Exception as exc:
                print(f"[ERROR] sitl_loop: {exc}")

    def mission_reply_loop(self):
        while self.running:
            try:
                packet, src_addr = self.sock_mission.recvfrom(65535)
                # Any packet arriving on this socket is treated as a Mission Control reply.
                # Do not require an exact source port match; some UDP stacks/tools choose
                # a different source port, and the old strict check could silently drop commands.
                if self.last_sitl_addr is not None:
                    self.sock_sitl.sendto(packet, self.last_sitl_addr)
                    self.total_returned += 1
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                self.running = False
            except Exception as exc:
                print(f"[ERROR] mission_reply_loop: {exc}")

    def delay_sender_loop(self):
        while self.running:
            now = time.time()
            ready = self.delay_queue.pop_ready(now)
            for _, packet in ready:
                self.sock_mission.sendto(packet, self.mission_addr)
                self.total_forwarded += 1
            time.sleep(0.001)

    def watchdog_loop(self):
        start_time = time.time()
        last_warning_time = 0.0
        last_count = -1
        while self.running:
            now = time.time()
            if self.total_in == 0 and now - start_time >= 5.0 and now - last_warning_time >= 5.0:
                print(
                    "[WARN] No telemetry packets have reached the proxy yet. "
                    f"Make sure MAVProxy/SITL is outputting UDP to port {self.listen_addr[1]} "
                    f"and Mission Control is listening on port {self.mission_addr[1]}."
                )
                last_warning_time = now
            elif self.total_in != last_count and now - last_warning_time >= 5.0:
                print(
                    f"[LIVE] in={self.total_in} forwarded={self.total_forwarded} "
                    f"dropped={self.total_dropped} returned={self.total_returned}"
                )
                last_count = self.total_in
                last_warning_time = now
            time.sleep(1.0)

    def start(self):
        self.running = True
        threading.Thread(target=self.sitl_loop, daemon=True).start()
        threading.Thread(target=self.mission_reply_loop, daemon=True).start()
        threading.Thread(target=self.delay_sender_loop, daemon=True).start()
        threading.Thread(target=self.watchdog_loop, daemon=True).start()

        print(f"[INFO] Listening for SITL/MAVProxy on UDP {self.listen_addr[0]}:{self.listen_addr[1]}")
        print(f"[INFO] Forwarding telemetry to Mission Control on UDP {self.mission_addr[0]}:{self.mission_addr[1]}")
        print(f"[INFO] Listening for Mission Control replies on UDP {self.mission_reply_addr[0]}:{self.mission_reply_addr[1]}")
        print(f"[INFO] Logging to {self.logger.path}")
        if self.risk_stream_enabled:
            print(f"[INFO] Streaming proxy/refiner input rows to Mission Control risk listener on UDP {self.risk_stream_addr[0]}:{self.risk_stream_addr[1]}")
        else:
            print("[INFO] Proxy risk stream is disabled")
        print(
            f"[INFO] Hotspot center=({self.center_x_m}, {self.center_y_m}) "
            f"radius={self.radius_m}m blackout_radius={self.blackout_radius_m}m"
        )

        try:
            while self.running:
                time.sleep(0.25)
        except KeyboardInterrupt:
            self.running = False
        finally:
            self.sock_sitl.close()
            self.sock_mission.close()
            self.sock_risk.close()
            self.logger.close()

    def print_stats(self):
        print(
            f"[STATS] in={self.total_in} "
            f"forwarded={self.total_forwarded} "
            f"returned={self.total_returned} "
            f"dropped={self.total_dropped}"
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="MAVLink hotspot proxy with per-packet .txt logging")
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--sitl-listen-port", type=int, default=14550)
    parser.add_argument("--mission-host", default="127.0.0.1")
    parser.add_argument("--mission-port", type=int, default=14551)
    parser.add_argument("--mission-reply-port", type=int, default=14552)
    parser.add_argument("--log-file", default="signal_loss_run.txt")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--risk-stream-host", default="127.0.0.1")
    parser.add_argument("--risk-stream-port", type=int, default=14600)
    parser.add_argument("--disable-risk-stream", action="store_true")

    parser.add_argument("--center-x", type=float, default=50.0)
    parser.add_argument("--center-y", type=float, default=25.0)
    parser.add_argument("--radius", type=float, default=100.0)
    parser.add_argument("--blackout-radius", type=float, default=0.0)
    parser.add_argument("--falloff-power", type=float, default=1.5)

    parser.add_argument("--max-random-loss-prob", type=float, default=0.0)
    parser.add_argument("--max-base-latency-s", type=float, default=0.0)
    parser.add_argument("--max-jitter-s", type=float, default=0.0)
    parser.add_argument("--max-keep-every-n", type=int, default=1)
    parser.add_argument("--max-burst-trigger-prob", type=float, default=0.0)
    parser.add_argument("--max-burst-duration-s", type=float, default=0.0)
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    proxy = BidirectionalLocalHotspotProxy(
        listen_host=args.listen_host,
        sitl_listen_port=args.sitl_listen_port,
        mission_host=args.mission_host,
        mission_port=args.mission_port,
        mission_reply_port=args.mission_reply_port,
        log_file=args.log_file,
        run_id=args.run_id,
        risk_stream_host=args.risk_stream_host,
        risk_stream_port=args.risk_stream_port,
        risk_stream_enabled=not args.disable_risk_stream,
    )

    proxy.center_x_m = args.center_x
    proxy.center_y_m = args.center_y
    proxy.radius_m = args.radius
    proxy.blackout_radius_m = args.blackout_radius
    proxy.falloff_power = args.falloff_power

    proxy.max_random_loss_prob = args.max_random_loss_prob
    proxy.max_base_latency_s = args.max_base_latency_s
    proxy.max_jitter_s = args.max_jitter_s
    proxy.max_keep_every_n = args.max_keep_every_n
    proxy.max_burst_trigger_prob = args.max_burst_trigger_prob
    proxy.max_burst_duration_s = args.max_burst_duration_s

    try:
        proxy.start()
    finally:
        proxy.print_stats()


if __name__ == "__main__":
    main()
