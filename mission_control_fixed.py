from __future__ import annotations

import math
import os
import time
from typing import List, Optional, Tuple

from pymavlink import mavutil

try:
    from live_risk_pipeline import ProxyRiskListener
except Exception as risk_import_exc:
    ProxyRiskListener = None
    RISK_IMPORT_ERROR = risk_import_exc
else:
    RISK_IMPORT_ERROR = None

# NOTE: If you are running through MAVProxy, the most reliable setup is to run
# `set streamrate -1` in the MAVProxy console so it does not keep overriding
# telemetry rates requested by this script.

# Connection / timing
CONNECTION_STRING = "udpin:0.0.0.0:14551"

# Live risk scoring side-channel from proxy_fixed.py. The MAVLink connection
# still carries flight telemetry/commands on 14551/14552; this UDP listener is
# separate so AI scoring never consumes MAVLink packets needed by flight logic.
RISK_STREAM_ENABLED = os.environ.get("RISK_STREAM_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
RISK_STREAM_HOST = os.environ.get("RISK_STREAM_HOST", "0.0.0.0")
RISK_STREAM_PORT = int(os.environ.get("RISK_STREAM_PORT", "14600"))
RISK_PRINT_PERIOD_S = float(os.environ.get("RISK_PRINT_PERIOD_S", "1.0"))
RISK_MODEL_PATH = os.environ.get("RISK_MODEL_PATH", "risk_model.json")

POSITION_READY_TIMEOUT_S = 90
MODE_CHANGE_TIMEOUT_S = 10
ARM_RETRY_COUNT = 5
ARM_RETRY_DELAY_S = 3
POST_ARM_DELAY_S = 2
MISSION_ALT_M = 10.0
GUIDED_TAKEOFF_ALT_M = MISSION_ALT_M
TAKEOFF_TIMEOUT_S = 60
RTL_DISARM_TIMEOUT_S = 240

# Guided waypoint behaviour
WAYPOINT_HORIZ_TOL_M = 2.5
WAYPOINT_VERT_TOL_M = 1.5
WAYPOINT_STABLE_TIME_S = 1.5
WAYPOINT_TIMEOUT_S = 150.0
WAYPOINT_ALT_M = MISSION_ALT_M
POSITION_STREAM_HZ = 5
POSITION_SAMPLE_TIMEOUT_S = 3.0
WAYPOINT_PROGRESS_EPSILON_M = 0.5
WAYPOINT_NO_PROGRESS_TIMEOUT_S = 10.0
WAYPOINT_RESEND_PERIOD_S = 6.0

# Build the route relative to the current vehicle position instead of hardcoding
# the default SITL home location.
WAYPOINT_OFFSETS = [
    (0.0002, 0.0002, WAYPOINT_ALT_M),
    (0.0004, 0.0004, WAYPOINT_ALT_M),
    (0.0006, 0.0002, WAYPOINT_ALT_M),
    (0.0008, 0.0000, WAYPOINT_ALT_M),
    (0.0010, -0.0002, WAYPOINT_ALT_M),
]

# EKF status bits from EKF_STATUS_FLAGS.
EKF_ATTITUDE = 1
EKF_VELOCITY_HORIZ = 2
EKF_POS_HORIZ_ABS = 16
EKF_POS_VERT_ABS = 32
EKF_CONST_POS_MODE = 128
EKF_UNINITIALIZED = 1024
EKF_GPS_GLITCHING = 32768

# Message IDs used when requesting telemetry.
MSG_ID_GPS_RAW_INT = 24
MSG_ID_LOCAL_POSITION_NED = 32
MSG_ID_GLOBAL_POSITION_INT = 33
MSG_ID_EKF_STATUS_REPORT = 193
MSG_ID_HOME_POSITION = 242

# Position target type-mask: ignore velocity, acceleration, yaw and yaw-rate.
# Only lat/lon/alt are used.
SET_POSITION_TYPEMASK_POSITION_ONLY = 3576

MAV_RESULT_NAMES = {
    0: "ACCEPTED",
    1: "TEMPORARILY_REJECTED",
    2: "DENIED",
    3: "UNSUPPORTED",
    4: "FAILED",
    5: "IN_PROGRESS",
    6: "CANCELLED",
}

connection = None
COMMAND_TARGET_COMPONENT = None
STREAM_TARGET_COMPONENT = 0
risk_listener = None


def start_risk_listener() -> None:
    global risk_listener
    if not RISK_STREAM_ENABLED:
        print("[RISK] Risk stream disabled by RISK_STREAM_ENABLED=0")
        return
    if ProxyRiskListener is None:
        print(f"[RISK WARN] live_risk_pipeline could not be imported: {RISK_IMPORT_ERROR}")
        return
    try:
        risk_listener = ProxyRiskListener(
            listen_host=RISK_STREAM_HOST,
            listen_port=RISK_STREAM_PORT,
            print_period_s=RISK_PRINT_PERIOD_S,
            model_path=RISK_MODEL_PATH,
        )
        risk_listener.start()
    except Exception as exc:
        risk_listener = None
        print(f"[RISK WARN] Could not start proxy risk listener: {exc}")


def stop_risk_listener() -> None:
    global risk_listener
    if risk_listener is not None:
        try:
            risk_listener.stop()
        except Exception as exc:
            print(f"[RISK WARN] Could not stop proxy risk listener cleanly: {exc}")
        risk_listener = None


def mav_result_name(result: Optional[int]) -> str:
    if result is None:
        return "NO_ACK"
    return MAV_RESULT_NAMES.get(int(result), str(result))


def monotonic_deadline(timeout_s: float) -> float:
    return time.monotonic() + timeout_s


def log_ap_text(text: str) -> None:
    print(f"[AP] {text}")


def connect() -> None:
    global connection, COMMAND_TARGET_COMPONENT
    print("Mission control starting")
    connection = mavutil.mavlink_connection(CONNECTION_STRING)
    connection.wait_heartbeat()
    print(
        f"Connected to system {connection.target_system}, "
        f"component {connection.target_component}"
    )

    # Some telemetry routers report component 0. Flight commands should still
    # target the autopilot component. Telemetry-rate requests are sent to
    # component 0 as recommended by the ArduPilot docs.
    if int(connection.target_component) == 0:
        COMMAND_TARGET_COMPONENT = mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1
        print(
            "Heartbeat reported component 0; using autopilot component "
            f"{COMMAND_TARGET_COMPONENT} for flight commands"
        )
    else:
        COMMAND_TARGET_COMPONENT = int(connection.target_component)



def request_position_stream(rate_hz: int = POSITION_STREAM_HZ) -> None:
    connection.mav.request_data_stream_send(
        connection.target_system,
        STREAM_TARGET_COMPONENT,
        mavutil.mavlink.MAV_DATA_STREAM_POSITION,
        int(rate_hz),
        1,
    )



def request_message_once(message_id: int) -> None:
    connection.mav.command_long_send(
        connection.target_system,
        STREAM_TARGET_COMPONENT,
        mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,
        0,
        message_id,
        0,
        0,
        0,
        0,
        0,
        0,
    )



def request_position_related_messages() -> None:
    request_position_stream(POSITION_STREAM_HZ)
    request_message_once(MSG_ID_LOCAL_POSITION_NED)
    request_message_once(MSG_ID_GLOBAL_POSITION_INT)
    request_message_once(MSG_ID_GPS_RAW_INT)
    request_message_once(MSG_ID_EKF_STATUS_REPORT)
    request_message_once(MSG_ID_HOME_POSITION)



def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_m = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2.0) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    )
    return 2.0 * earth_radius_m * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))



def ekf_position_is_good(flags: int) -> bool:
    return bool(
        (flags & EKF_ATTITUDE)
        and (flags & EKF_VELOCITY_HORIZ)
        and (flags & EKF_POS_HORIZ_ABS)
        and (flags & EKF_POS_VERT_ABS)
        and not (flags & EKF_CONST_POS_MODE)
        and not (flags & EKF_UNINITIALIZED)
        and not (flags & EKF_GPS_GLITCHING)
    )



def wait_for_command_ack(command_id: int, timeout_s: float = 10.0) -> Optional[int]:
    deadline = monotonic_deadline(timeout_s)
    while time.monotonic() < deadline:
        msg = connection.recv_match(
            type=["COMMAND_ACK", "STATUSTEXT"],
            blocking=True,
            timeout=1,
        )
        if msg is None:
            continue

        msg_type = msg.get_type()
        if msg_type == "STATUSTEXT":
            log_ap_text(msg.text)
            continue

        if int(getattr(msg, "command", -1)) == command_id:
            result = int(getattr(msg, "result", -1))
            print(
                f"COMMAND_ACK command={command_id} result="
                f"{result} ({mav_result_name(result)})"
            )
            return result

    return None



def wait_for_position_ready(
    timeout_s: float = POSITION_READY_TIMEOUT_S,
) -> Tuple[float, float, float]:
    print("Waiting for GPS + EKF position estimate")
    request_position_related_messages()

    have_fix = False
    have_global_position = False
    saw_ekf_status = False
    ekf_ok = False

    current_lat = None
    current_lon = None
    current_rel_alt_m = None

    next_poll_time = time.monotonic()
    deadline = monotonic_deadline(timeout_s)
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_poll_time:
            request_message_once(MSG_ID_LOCAL_POSITION_NED)
            request_message_once(MSG_ID_GPS_RAW_INT)
            request_message_once(MSG_ID_EKF_STATUS_REPORT)
            request_message_once(MSG_ID_GLOBAL_POSITION_INT)
            request_message_once(MSG_ID_HOME_POSITION)
            next_poll_time = now + 1.5

        msg = connection.recv_match(
            type=[
                "GPS_RAW_INT",
                "GLOBAL_POSITION_INT",
                "EKF_STATUS_REPORT",
                "HOME_POSITION",
                "STATUSTEXT",
                "COMMAND_ACK",
                "HEARTBEAT",
            ],
            blocking=True,
            timeout=0.5,
        )
        if msg is None:
            continue

        msg_type = msg.get_type()

        if msg_type == "STATUSTEXT":
            log_ap_text(msg.text)
            continue

        if msg_type == "COMMAND_ACK":
            # REQUEST_MESSAGE ACKs are expected and not useful here.
            continue

        if msg_type == "GPS_RAW_INT":
            fix_type = int(getattr(msg, "fix_type", 0))
            satellites_visible = int(getattr(msg, "satellites_visible", 0))
            have_fix = fix_type >= 3
            print(f"GPS fix_type={fix_type} satellites={satellites_visible}")
            continue

        if msg_type == "GLOBAL_POSITION_INT":
            current_lat = msg.lat / 1e7
            current_lon = msg.lon / 1e7
            current_rel_alt_m = msg.relative_alt / 1000.0
            have_global_position = not (msg.lat == 0 and msg.lon == 0)
            print(
                f"Global position lat={current_lat:.7f}, lon={current_lon:.7f}, "
                f"rel_alt={current_rel_alt_m:.2f} m"
            )
            continue

        if msg_type == "EKF_STATUS_REPORT":
            saw_ekf_status = True
            flags = int(getattr(msg, "flags", 0))
            ekf_ok = ekf_position_is_good(flags)
            print(f"EKF flags={flags} ekf_ok={ekf_ok}")
            continue

        if msg_type == "HOME_POSITION":
            home_lat = msg.latitude / 1e7
            home_lon = msg.longitude / 1e7
            home_alt_m = msg.altitude / 1000.0
            print(
                f"HOME_POSITION lat={home_lat:.7f}, lon={home_lon:.7f}, "
                f"alt={home_alt_m:.2f} m"
            )
            continue

        if have_fix and have_global_position and ekf_ok:
            break

    if not (have_fix and have_global_position and ekf_ok):
        raise TimeoutError(
            "Position estimate did not become ready in time. "
            f"have_fix={have_fix}, have_global_position={have_global_position}, "
            f"saw_ekf_status={saw_ekf_status}, ekf_ok={ekf_ok}"
        )

    if current_lat is None or current_lon is None or current_rel_alt_m is None:
        raise TimeoutError("No usable GLOBAL_POSITION_INT received")

    print("Position estimate ready")
    time.sleep(2)
    return current_lat, current_lon, current_rel_alt_m



def set_home_to_current_position() -> None:
    print("Requesting home position = current position")
    connection.mav.command_long_send(
        connection.target_system,
        COMMAND_TARGET_COMPONENT,
        mavutil.mavlink.MAV_CMD_DO_SET_HOME,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    result = wait_for_command_ack(mavutil.mavlink.MAV_CMD_DO_SET_HOME, timeout_s=5)
    if result not in (None, mavutil.mavlink.MAV_RESULT_ACCEPTED):
        print(
            "Home set command was not explicitly accepted; continuing anyway. "
            f"result={mav_result_name(result)}"
        )
    request_message_once(MSG_ID_HOME_POSITION)



def wait_for_mode(mode_name: str, timeout_s: float = MODE_CHANGE_TIMEOUT_S) -> None:
    mode_map = connection.mode_mapping()
    if mode_map is None or mode_name not in mode_map:
        raise ValueError(f"Unknown mode: {mode_name}")

    expected_mode_id = int(mode_map[mode_name])
    deadline = monotonic_deadline(timeout_s)

    while time.monotonic() < deadline:
        msg = connection.recv_match(
            type=["HEARTBEAT", "STATUSTEXT"],
            blocking=True,
            timeout=1,
        )
        if msg is None:
            continue

        if msg.get_type() == "STATUSTEXT":
            log_ap_text(msg.text)
            continue

        current_mode = int(getattr(msg, "custom_mode", -1))
        if current_mode == expected_mode_id:
            print(f"Mode confirmed: {mode_name}")
            return

    raise TimeoutError(f"Timed out waiting for mode {mode_name}")



def set_mode(mode_name: str) -> None:
    mode_map = connection.mode_mapping()
    if mode_map is None or mode_name not in mode_map:
        raise ValueError(f"Unknown mode: {mode_name}")

    mode_id = int(mode_map[mode_name])
    connection.mav.set_mode_send(
        connection.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id,
    )
    print(f"Requested mode change to {mode_name}")
    wait_for_mode(mode_name)



def arm() -> None:
    connection.mav.command_long_send(
        connection.target_system,
        COMMAND_TARGET_COMPONENT,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    print("Sent arm command")



def wait_for_arm_ack(timeout_s: float = 15.0) -> Tuple[Optional[int], Optional[str]]:
    deadline = monotonic_deadline(timeout_s)
    last_text = None

    while time.monotonic() < deadline:
        msg = connection.recv_match(
            type=["COMMAND_ACK", "STATUSTEXT", "HEARTBEAT"],
            blocking=True,
            timeout=1,
        )
        if msg is None:
            continue

        msg_type = msg.get_type()

        if msg_type == "STATUSTEXT":
            last_text = str(msg.text)
            log_ap_text(last_text)
            continue

        if (
            msg_type == "COMMAND_ACK"
            and int(getattr(msg, "command", -1))
            == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
        ):
            result = int(getattr(msg, "result", -1))
            print(f"Arm COMMAND_ACK result={result} ({mav_result_name(result)})")
            return result, last_text

        if msg_type == "HEARTBEAT":
            armed = bool(
                int(getattr(msg, "base_mode", 0))
                & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            )
            if armed:
                print("Vehicle is armed")
                return mavutil.mavlink.MAV_RESULT_ACCEPTED, last_text

    return None, last_text



def arm_with_retries(
    retries: int = ARM_RETRY_COUNT,
    delay_s: float = ARM_RETRY_DELAY_S,
) -> None:
    last_reason = None

    for attempt in range(1, retries + 1):
        print(f"Arm attempt {attempt}/{retries}")
        arm()
        arm_result, last_reason = wait_for_arm_ack()

        if arm_result in (
            mavutil.mavlink.MAV_RESULT_ACCEPTED,
            mavutil.mavlink.MAV_RESULT_IN_PROGRESS,
        ):
            print("Arming accepted")
            return

        print(
            "Arming not accepted yet, waiting "
            f"{delay_s} seconds before retry"
        )
        time.sleep(delay_s)

    reason_suffix = f" Last reported reason: {last_reason}" if last_reason else ""
    raise RuntimeError(
        "Vehicle failed to arm after multiple attempts." + reason_suffix
    )



def guided_takeoff(altitude_m: float) -> None:
    connection.mav.command_long_send(
        connection.target_system,
        COMMAND_TARGET_COMPONENT,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        altitude_m,
    )
    print(f"Sent GUIDED takeoff command to {altitude_m:.1f} m")

    result = wait_for_command_ack(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, timeout_s=5)
    if result not in (None, mavutil.mavlink.MAV_RESULT_ACCEPTED):
        raise RuntimeError(
            "Takeoff command rejected with result="
            f"{mav_result_name(result)}"
        )



def wait_for_global_position_sample(
    timeout_s: float = POSITION_SAMPLE_TIMEOUT_S,
) -> Optional[Tuple[float, float, float]]:
    # Passive wait first; if nothing arrives soon, actively poll once.
    request_deadline = time.monotonic() + min(1.0, timeout_s / 2.0)
    deadline = monotonic_deadline(timeout_s)
    requested_once = False

    while time.monotonic() < deadline:
        now = time.monotonic()
        if not requested_once and now >= request_deadline:
            request_message_once(MSG_ID_GLOBAL_POSITION_INT)
            requested_once = True

        msg = connection.recv_match(
            type=["GLOBAL_POSITION_INT", "STATUSTEXT", "COMMAND_ACK", "HEARTBEAT"],
            blocking=True,
            timeout=0.5,
        )
        if msg is None:
            continue

        msg_type = msg.get_type()
        if msg_type == "STATUSTEXT":
            log_ap_text(msg.text)
            continue
        if msg_type == "COMMAND_ACK":
            continue
        if msg_type == "HEARTBEAT":
            continue

        return (
            msg.lat / 1e7,
            msg.lon / 1e7,
            msg.relative_alt / 1000.0,
        )

    return None



def wait_until_altitude_reached(
    target_alt_m: float,
    tolerance_m: float = 0.7,
    timeout_s: float = TAKEOFF_TIMEOUT_S,
) -> None:
    request_position_stream(POSITION_STREAM_HZ)
    deadline = monotonic_deadline(timeout_s)

    while time.monotonic() < deadline:
        sample = wait_for_global_position_sample(timeout_s=2.5)
        if sample is None:
            print("No fresh GLOBAL_POSITION_INT while climbing; polling again")
            continue

        rel_alt_m = sample[2]
        print(f"Relative altitude: {rel_alt_m:.2f} m")

        if rel_alt_m >= (target_alt_m - tolerance_m):
            print(f"Reached takeoff altitude: {rel_alt_m:.2f} m")
            return

    raise TimeoutError(f"Did not reach {target_alt_m:.1f} m in time")



def send_guided_waypoint(lat: float, lon: float, alt_m: float) -> None:
    connection.mav.set_position_target_global_int_send(
        0,
        connection.target_system,
        COMMAND_TARGET_COMPONENT,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        SET_POSITION_TYPEMASK_POSITION_ONLY,
        int(lat * 1e7),
        int(lon * 1e7),
        float(alt_m),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    print(f"Sent GUIDED waypoint lat={lat:.7f}, lon={lon:.7f}, alt={alt_m:.1f} m")



def wait_until_waypoint_reached(
    target_lat: float,
    target_lon: float,
    target_alt_m: float,
    horiz_tol_m: float = WAYPOINT_HORIZ_TOL_M,
    vert_tol_m: float = WAYPOINT_VERT_TOL_M,
    stable_time_s: float = WAYPOINT_STABLE_TIME_S,
    timeout_s: float = WAYPOINT_TIMEOUT_S,
) -> None:
    request_position_stream(POSITION_STREAM_HZ)

    deadline = monotonic_deadline(timeout_s)
    inside_since = None
    last_report_time = 0.0
    last_distance_m = None
    last_progress_time = time.monotonic()
    last_resend_time = time.monotonic()

    while time.monotonic() < deadline:
        sample = wait_for_global_position_sample(timeout_s=POSITION_SAMPLE_TIMEOUT_S)
        now = time.monotonic()

        if sample is None:
            print("No fresh GLOBAL_POSITION_INT; polling again")
            if now - last_resend_time >= WAYPOINT_RESEND_PERIOD_S:
                print("Re-sending current waypoint after telemetry gap")
                send_guided_waypoint(target_lat, target_lon, target_alt_m)
                last_resend_time = now
            continue

        cur_lat, cur_lon, cur_alt_m = sample
        dist_m = haversine_m(cur_lat, cur_lon, target_lat, target_lon)
        alt_err_m = abs(cur_alt_m - target_alt_m)

        if last_distance_m is None or dist_m < (last_distance_m - WAYPOINT_PROGRESS_EPSILON_M):
            last_progress_time = now
            last_distance_m = dist_m
        elif last_distance_m is None:
            last_distance_m = dist_m

        if now - last_report_time >= 1.0:
            print(
                f"Waypoint check: dist={dist_m:.2f} m, "
                f"rel_alt={cur_alt_m:.2f} m, alt_err={alt_err_m:.2f} m"
            )
            last_report_time = now

        if dist_m <= horiz_tol_m and alt_err_m <= vert_tol_m:
            if inside_since is None:
                inside_since = now
            elif (now - inside_since) >= stable_time_s:
                print("Waypoint reached")
                return
        else:
            inside_since = None

        if (
            now - last_progress_time >= WAYPOINT_NO_PROGRESS_TIMEOUT_S
            or now - last_resend_time >= WAYPOINT_RESEND_PERIOD_S
        ):
            print("Waypoint progress stalled; re-sending current waypoint")
            send_guided_waypoint(target_lat, target_lon, target_alt_m)
            last_progress_time = now
            last_resend_time = now

    raise TimeoutError(
        f"Did not reach waypoint lat={target_lat:.7f}, lon={target_lon:.7f}, alt={target_alt_m:.1f}"
    )



def rtl() -> None:
    set_mode("RTL")
    print("Requested RTL")



def wait_until_disarmed(timeout_s: float = RTL_DISARM_TIMEOUT_S) -> None:
    deadline = monotonic_deadline(timeout_s)

    while time.monotonic() < deadline:
        msg = connection.recv_match(
            type=["HEARTBEAT", "STATUSTEXT"],
            blocking=True,
            timeout=1,
        )
        if msg is None:
            continue

        if msg.get_type() == "STATUSTEXT":
            log_ap_text(msg.text)
            continue

        armed = bool(
            int(getattr(msg, "base_mode", 0))
            & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        )
        if not armed:
            print("Vehicle disarmed")
            return

    print("Timed out waiting for automatic disarm after RTL")



def build_guided_waypoints(base_lat: float, base_lon: float) -> List[Tuple[float, float, float]]:
    return [
        (base_lat + dlat, base_lon + dlon, alt)
        for dlat, dlon, alt in WAYPOINT_OFFSETS
    ]



def main() -> None:
    connect()
    start_risk_listener()

    try:
        current_lat, current_lon, _ = wait_for_position_ready()
        set_home_to_current_position()

        guided_waypoints = build_guided_waypoints(current_lat, current_lon)
        print("Step-by-step GUIDED route:")
        for index, (lat, lon, alt) in enumerate(guided_waypoints, start=1):
            print(f"  {index}: lat={lat:.7f}, lon={lon:.7f}, alt={alt:.1f} m")

        set_mode("GUIDED")
        arm_with_retries()
        time.sleep(POST_ARM_DELAY_S)

        print("About to send GUIDED takeoff")
        guided_takeoff(GUIDED_TAKEOFF_ALT_M)

        print("Waiting to reach takeoff altitude")
        wait_until_altitude_reached(GUIDED_TAKEOFF_ALT_M)
        time.sleep(2)

        print("Starting step-by-step GUIDED route")
        for i, (lat, lon, alt) in enumerate(guided_waypoints, start=1):
            print(f"Sending waypoint {i}/{len(guided_waypoints)}")
            send_guided_waypoint(lat, lon, alt)
            wait_until_waypoint_reached(lat, lon, alt)
            time.sleep(1)

        print("All guided waypoints completed, returning home")
        rtl()
        wait_until_disarmed()
        print("Mission complete")

    except Exception as exc:
        print(f"Mission aborted: {exc}")
        try:
            rtl()
        except Exception as rtl_exc:
            print(f"Unable to switch to RTL after failure: {rtl_exc}")
        raise
    finally:
        stop_risk_listener()


if __name__ == "__main__":
    main()
