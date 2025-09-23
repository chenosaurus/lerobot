import contextlib
import os
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None  # optional dependency

from lerobot.robots.lekiwi.lekiwi_remote_client import (
    LeKiwiRemoteClient,
    LeKiwiRemoteClientConfig,
)
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data


FPS = 30


def main():
    # Load environment variables from the repo root .env if available
    if load_dotenv is not None:
        repo_root = Path(__file__).resolve().parents[2]
        env_path = repo_root / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)

    livekit_url = os.getenv("LIVEKIT_URL", "")
    livekit_token = os.getenv("LIVEKIT_LEADER_TOKEN", "")
    if not livekit_url or not livekit_token:
        raise ValueError(
            "LIVEKIT_URL and LIVEKIT_TOKEN environment variables must be set to use the remote client."
        )

    # Configure robot and teleoperators
    robot_config = LeKiwiRemoteClientConfig(livekit_url=livekit_url, livekit_token=livekit_token, id="lekiwi")
    leader_port = os.getenv("SO100_LEADER_PORT", "/dev/tty.usbmodem58FA1025471")
    teleop_arm_config = SO100LeaderConfig(port=leader_port, id="l1")
    keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

    robot = LeKiwiRemoteClient(robot_config)
    leader_arm = SO100Leader(teleop_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    # Connect all devices
    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    _init_rerun(session_name="lekiwi_remote_teleop")

    if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
        raise ValueError("Robot, leader arm or keyboard is not connected!")

    # Base teleop key mapping and speed control
    teleop_keys = {
        "forward": "w",
        "backward": "s",
        "left": "a",
        "right": "d",
        "rotate_left": "z",
        "rotate_right": "x",
        "speed_up": "r",
        "speed_down": "f",
        "quit": "q",
    }
    speed_levels = [
        {"xy": 0.1, "theta": 30},  # slow
        {"xy": 0.2, "theta": 60},  # medium
        {"xy": 0.3, "theta": 90},  # fast
    ]
    speed_index = 0

    try:
        while True:
            t0 = time.perf_counter()

            observation = robot.get_observation()

            # Arm action from SO-100 leader
            arm_action = leader_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}

            # Base action from keyboard
            keyboard_keys = keyboard.get_action()
            pressed = set(keyboard_keys.keys())

            # Update speed level
            if teleop_keys["speed_up"] in pressed:
                speed_index = min(speed_index + 1, len(speed_levels) - 1)
            if teleop_keys["speed_down"] in pressed:
                speed_index = max(speed_index - 1, 0)

            xy_speed = speed_levels[speed_index]["xy"]
            theta_speed = speed_levels[speed_index]["theta"]

            x_cmd = 0.0
            y_cmd = 0.0
            theta_cmd = 0.0

            if teleop_keys["forward"] in pressed:
                x_cmd += xy_speed
            if teleop_keys["backward"] in pressed:
                x_cmd -= xy_speed
            if teleop_keys["left"] in pressed:
                y_cmd += xy_speed
            if teleop_keys["right"] in pressed:
                y_cmd -= xy_speed
            if teleop_keys["rotate_left"] in pressed:
                theta_cmd += theta_speed
            if teleop_keys["rotate_right"] in pressed:
                theta_cmd -= theta_speed

            base_action = {
                "x.vel": x_cmd,
                "y.vel": y_cmd,
                "theta.vel": theta_cmd,
            }

            # Log and send action
            combined_action = {**arm_action, **base_action}
            log_rerun_data(observation, combined_action)
            robot.send_action(combined_action)

            # Quit condition
            if teleop_keys["quit"] in pressed:
                break

            busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    finally:
        # Best-effort disconnect
        with contextlib.suppress(Exception):
            robot.disconnect()
        with contextlib.suppress(Exception):
            leader_arm.disconnect()
        with contextlib.suppress(Exception):
            keyboard.disconnect()


if __name__ == "__main__":
    main()


