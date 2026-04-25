import socket
import threading
import time
import random
import heapq
from math import hypot

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


class BidirectionalLocalHotspotProxy:
    """
    MAVProxy/SITL -> proxy(14560) -> Mission Control(14550)
    Mission Control replies -> proxy(14551) -> MAVProxy/SITL source port

    The SITL -> Mission Control direction can be impaired.
    The Mission Control -> SITL direction is passed through unchanged so
    mission uploads and command packets are not broken by the proxy itself.
    """

    def __init__(
        self,
        listen_host="127.0.0.1",
        sitl_listen_port=14560,
        mission_host="127.0.0.1",
        mission_port=14550,
        mission_reply_port=14551,
    ):
        self.listen_addr = (listen_host, sitl_listen_port)
        self.mission_addr = (mission_host, mission_port)
        self.mission_reply_addr = (listen_host, mission_reply_port)

        # Receives telemetry from MAVProxy/SITL and also sends commands back.
        self.sock_sitl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_sitl.bind(self.listen_addr)
        self.sock_sitl.settimeout(0.1)

        # Sends telemetry to Mission Control from a fixed source port and
        # receives Mission Control replies on that same bound port.
        self.sock_mission = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_mission.bind(self.mission_reply_addr)
        self.sock_mission.settimeout(0.1)

        self.delay_queue = DelayQueue()
        self.mav_parser = mavlink2.MAVLink(None)

        self.running = False
        self.last_sitl_addr = None

        self.total_in = 0
        self.total_forwarded = 0
        self.total_dropped = 0
        self.total_returned = 0

        # Local position tracking.
        self.pos_valid = False
        self.x_m = 0.0
        self.y_m = 0.0
        self.z_m = 0.0

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
        for byte_value in packet:
            try:
                msg = self.mav_parser.parse_char(bytes([byte_value]))
                if msg is None:
                    continue
                if msg.get_type() == "LOCAL_POSITION_NED":
                    self.x_m = msg.x
                    self.y_m = msg.y
                    self.z_m = msg.z
                    self.pos_valid = True
            except Exception:
                pass

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

    def process_sitl_packet(self, packet):
        self.update_position_from_packet(packet)

        now = time.time()
        distance_m = self.distance_to_hotspot()
        strength = self.compute_strength(distance_m)

        if self.blackout_active(distance_m):
            self.total_dropped += 1
            return

        if self.burst_active(now, strength):
            self.total_dropped += 1
            return

        if self.drop_by_random_loss(strength):
            self.total_dropped += 1
            return

        if self.drop_by_throttle(strength):
            self.total_dropped += 1
            return

        delay = self.compute_delay(strength)
        if delay > 0:
            self.delay_queue.push(now + delay, packet)
        else:
            self.sock_mission.sendto(packet, self.mission_addr)
            self.total_forwarded += 1

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
                # Ignore packets from anything except Mission Control.
                if src_addr[0] != self.mission_addr[0] or src_addr[1] != self.mission_addr[1]:
                    continue
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

    def start(self):
        self.running = True
        threading.Thread(target=self.sitl_loop, daemon=True).start()
        threading.Thread(target=self.mission_reply_loop, daemon=True).start()
        threading.Thread(target=self.delay_sender_loop, daemon=True).start()

        print(f"[INFO] Listening for SITL/MAVProxy on UDP {self.listen_addr[0]}:{self.listen_addr[1]}")
        print(f"[INFO] Forwarding telemetry to Mission Control on UDP {self.mission_addr[0]}:{self.mission_addr[1]}")
        print(f"[INFO] Listening for Mission Control replies on UDP {self.mission_reply_addr[0]}:{self.mission_reply_addr[1]}")
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

    def print_stats(self):
        print(
            f"[STATS] in={self.total_in} "
            f"forwarded={self.total_forwarded} "
            f"returned={self.total_returned} "
            f"dropped={self.total_dropped}"
        )


if __name__ == "__main__":
    proxy = BidirectionalLocalHotspotProxy(
        listen_host="127.0.0.1",
        sitl_listen_port=14560,
        mission_host="127.0.0.1",
        mission_port=14550,
        mission_reply_port=14551,
    )

    # Keep these OFF for the first validation run.
    proxy.center_x_m = 50.0
    proxy.center_y_m = 25.0
    proxy.radius_m = 100.0
    proxy.blackout_radius_m = 0.0
    proxy.falloff_power = 1.5

    proxy.max_random_loss_prob = 0.0
    proxy.max_base_latency_s = 0.0
    proxy.max_jitter_s = 0.0
    proxy.max_keep_every_n = 1
    proxy.max_burst_trigger_prob = 0.0
    proxy.max_burst_duration_s = 0.0

    try:
        proxy.start()
    finally:
        proxy.print_stats()
