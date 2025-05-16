# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains logic to instantiate a robot, read information from its motors and cameras,
and send orders to its motors.
"""
# TODO(rcadene, aliberts): reorganize the codebase into one file per robot, with the associated
# calibration procedure, to make it easy for people to add their own robot.

import json
import logging
import time
import warnings
from pathlib import Path
import threading

import numpy as np
import torch

# Livekit imports
import asyncio
from livekit import rtc


from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.motors.utils import MotorsBus, make_motors_buses_from_configs
from lerobot.common.robot_devices.robots.configs import ManipulatorRobotConfig
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError


def ensure_safe_goal_position(
    goal_pos: torch.Tensor, present_pos: torch.Tensor, max_relative_target: float | list[float]
):
    # Cap relative action target magnitude for safety.
    diff = goal_pos - present_pos
    max_relative_target = torch.tensor(max_relative_target)
    safe_diff = torch.minimum(diff, max_relative_target)
    safe_diff = torch.maximum(safe_diff, -max_relative_target)
    safe_goal_pos = present_pos + safe_diff

    if not torch.allclose(goal_pos, safe_goal_pos):
        logging.warning(
            "Relative goal position magnitude had to be clamped to be safe.\n"
            f"  requested relative goal position target: {diff}\n"
            f"    clamped relative goal position target: {safe_diff}"
        )

    return safe_goal_pos


class ManipulatorRobot:
    # TODO(rcadene): Implement force feedback
    """This class allows to control any manipulator robot of various number of motors.

    Non exhaustive list of robots:
    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot), with and without the wrist-to-elbow expansion, developed
    by Alexander Koch from [Tau Robotics](https://tau-robotics.com)
    - [Koch v1.1](https://github.com/jess-moss/koch-v1-1) developed by Jess Moss
    - [Aloha](https://www.trossenrobotics.com/aloha-kits) developed by Trossen Robotics

    Example of instantiation, a pre-defined robot config is required:
    ```python
    robot = ManipulatorRobot(KochRobotConfig())
    ```

    Example of overwriting motors during instantiation:
    ```python
    # Defines how to communicate with the motors of the leader and follower arms
    leader_arms = {
        "main": DynamixelMotorsBusConfig(
            port="/dev/tty.usbmodem575E0031751",
            motors={
                # name: (index, model)
                "shoulder_pan": (1, "xl330-m077"),
                "shoulder_lift": (2, "xl330-m077"),
                "elbow_flex": (3, "xl330-m077"),
                "wrist_flex": (4, "xl330-m077"),
                "wrist_roll": (5, "xl330-m077"),
                "gripper": (6, "xl330-m077"),
            },
        ),
    }
    follower_arms = {
        "main": DynamixelMotorsBusConfig(
            port="/dev/tty.usbmodem575E0032081",
            motors={
                # name: (index, model)
                "shoulder_pan": (1, "xl430-w250"),
                "shoulder_lift": (2, "xl430-w250"),
                "elbow_flex": (3, "xl330-m288"),
                "wrist_flex": (4, "xl330-m288"),
                "wrist_roll": (5, "xl330-m288"),
                "gripper": (6, "xl330-m288"),
            },
        ),
    }
    robot_config = KochRobotConfig(leader_arms=leader_arms, follower_arms=follower_arms)
    robot = ManipulatorRobot(robot_config)
    ```

    Example of overwriting cameras during instantiation:
    ```python
    # Defines how to communicate with 2 cameras connected to the computer.
    # Here, the webcam of the laptop and the phone (connected in USB to the laptop)
    # can be reached respectively using the camera indices 0 and 1. These indices can be
    # arbitrary. See the documentation of `OpenCVCamera` to find your own camera indices.
    cameras = {
        "laptop": OpenCVCamera(camera_index=0, fps=30, width=640, height=480),
        "phone": OpenCVCamera(camera_index=1, fps=30, width=640, height=480),
    }
    robot = ManipulatorRobot(KochRobotConfig(cameras=cameras))
    ```

    Once the robot is instantiated, connect motors buses and cameras if any (Required):
    ```python
    robot.connect()
    ```

    Example of highest frequency teleoperation, which doesn't require cameras:
    ```python
    while True:
        robot.teleop_step()
    ```

    Example of highest frequency data collection from motors and cameras (if any):
    ```python
    while True:
        observation, action = robot.teleop_step(record_data=True)
    ```

    Example of controlling the robot with a policy:
    ```python
    while True:
        # Uses the follower arms and cameras to capture an observation
        observation = robot.capture_observation()

        # Assumes a policy has been instantiated
        with torch.inference_mode():
            action = policy.select_action(observation)

        # Orders the robot to move
        robot.send_action(action)
    ```

    Example of disconnecting which is not mandatory since we disconnect when the object is deleted:
    ```python
    robot.disconnect()
    ```
    """

    def __init__(
        self,
        config: ManipulatorRobotConfig,
    ):
        self.config = config
        self.robot_type = self.config.type
        self.calibration_dir = Path(self.config.calibration_dir)
        self.leader_arms = make_motors_buses_from_configs(self.config.leader_arms)
        self.follower_arms = make_motors_buses_from_configs(self.config.follower_arms)
        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.is_connected = False
        self.logs = {}
        # Livekit room connection (for teleop leader)
        self.livekit_room = None
        self._livekit_loop = None
        self._livekit_thread = None
        self.is_teleop_leader = getattr(self.config, "is_teleop_leader", False)
        self.is_teleop_follower = getattr(self.config, "is_teleop_follower", False)

    def get_motor_names(self, arm: dict[str, MotorsBus]) -> list:
        return [f"{arm}_{motor}" for arm, bus in arm.items() for motor in bus.motors]

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        action_names = self.get_motor_names(self.leader_arms)
        state_names = self.get_motor_names(self.leader_arms)
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def available_arms(self):
        available_arms = []
        for name in self.follower_arms:
            arm_id = get_arm_id(name, "follower")
            available_arms.append(arm_id)
        for name in self.leader_arms:
            arm_id = get_arm_id(name, "leader")
            available_arms.append(arm_id)
        return available_arms

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        if not self.leader_arms and not self.follower_arms and not self.cameras:
            raise ValueError(
                "ManipulatorRobot doesn't have any device to connect. See example of usage in docstring of the class."
            )

        # Connect the arms
        for name in self.follower_arms:
            print(f"Connecting {name} follower arm.")
            self.follower_arms[name].connect()
        for name in self.leader_arms:
            print(f"Connecting {name} leader arm.")
            self.leader_arms[name].connect()

        if self.robot_type in ["koch", "koch_bimanual", "aloha"]:
            from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
        elif self.robot_type in ["so100", "so101", "moss", "lekiwi", "so100bimanual", "so100bimanual_teleop_follower", "so100bimanual_teleop_leader"]:
            from lerobot.common.robot_devices.motors.feetech import TorqueMode

        # Log available arms for debugging
        if self.is_teleop_follower:
            logging.info(f"Teleop follower has arms: {list(self.follower_arms.keys())}")

        # We assume that at connection time, arms are in a rest position, and torque can
        # be safely disabled to run calibration and/or set robot preset configurations.
        for name in self.follower_arms:
            self.follower_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
        for name in self.leader_arms:
            self.leader_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)

        self.activate_calibration()

        # Set robot preset (e.g. torque in leader gripper for Koch v1.1)
        if self.robot_type in ["koch", "koch_bimanual"]:
            self.set_koch_robot_preset()
        elif self.robot_type == "aloha":
            self.set_aloha_robot_preset()
        elif self.robot_type in ["so100", "so101", "moss", "lekiwi"]:
            self.set_so100_robot_preset()

        # Enable torque on all motors of the follower arms
        for name in self.follower_arms:
            print(f"Activating torque on {name} follower arm.")
            self.follower_arms[name].write("Torque_Enable", 1)

        if self.config.gripper_open_degree is not None:
            if self.robot_type not in ["koch", "koch_bimanual"]:
                raise NotImplementedError(
                    f"{self.robot_type} does not support position AND current control in the handle, which is require to set the gripper open."
                )
            # Set the leader arm in torque mode with the gripper motor set to an angle. This makes it possible
            # to squeeze the gripper and have it spring back to an open position on its own.
            for name in self.leader_arms:
                self.leader_arms[name].write("Torque_Enable", 1, "gripper")
                self.leader_arms[name].write("Goal_Position", self.config.gripper_open_degree, "gripper")

        # Check both arms can be read
        for name in self.follower_arms:
            self.follower_arms[name].read("Present_Position")
        for name in self.leader_arms:
            self.leader_arms[name].read("Present_Position")

        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()

        self.is_connected = True

        # Livekit connection for teleop leader
        if self.is_teleop_leader and rtc is not None:
            url = getattr(self.config, "livekit_url", None)
            token = getattr(self.config, "livekit_token", None)
            if url and token:
                try:
                    self._livekit_loop = asyncio.new_event_loop()
                    def run_loop(loop):
                        asyncio.set_event_loop(loop)
                        loop.run_forever()
                    self._livekit_thread = threading.Thread(target=run_loop, args=(self._livekit_loop,), daemon=True)
                    self._livekit_thread.start()
                    # Connect to Livekit room in the new loop
                    fut = asyncio.run_coroutine_threadsafe(
                        self._livekit_connect(url, token), self._livekit_loop
                    )
                    fut.result(timeout=10)
                    logging.info(f"Connected to Livekit room at {url}")
                except Exception as e:
                    logging.error(f"Failed to connect to Livekit: {e}")
                    self.livekit_room = None
            else:
                logging.warning("Livekit URL or token not provided in config; skipping Livekit connection.")
        # Livekit connection for teleop follower
        elif self.is_teleop_follower and rtc is not None:
            url = getattr(self.config, "livekit_url", None)
            token = getattr(self.config, "livekit_token", None)
            if url and token:
                try:
                    self._livekit_loop = asyncio.new_event_loop()
                    def run_loop(loop):
                        asyncio.set_event_loop(loop)
                        loop.run_forever()
                    self._livekit_thread = threading.Thread(target=run_loop, args=(self._livekit_loop,), daemon=True)
                    self._livekit_thread.start()
                    
                    # Connect to Livekit room in the new loop
                    fut = asyncio.run_coroutine_threadsafe(
                        self._livekit_connect_follower(url, token), self._livekit_loop
                    )
                    fut.result(timeout=10)
                    logging.info(f"Connected to Livekit room at {url} as teleop follower")
                except Exception as e:
                    logging.error(f"Failed to connect to Livekit as follower: {e}")
                    self.livekit_room = None
            else:
                logging.warning("Livekit URL or token not provided in config; skipping Livekit connection.")

    async def _livekit_connect(self, url, token):
        self.livekit_room = rtc.Room(loop=self._livekit_loop)
        await self.livekit_room.connect(url, token)

    async def _livekit_connect_follower(self, url, token):
        self.livekit_room = rtc.Room(loop=self._livekit_loop)
        
        # Handle incoming data packets from leader - using a synchronous callback
        @self.livekit_room.on("data_received")
        def on_data_received(data: rtc.DataPacket):
            if data.topic == "leader_arm_positions":
                # Decode and parse the data
                decoded_data = data.data.decode('utf-8')
                
                # Try to parse as JSON
                try:
                    json_data = json.loads(decoded_data)
                    print(f"Received data from leader: {json_data}")
                    # Use run_coroutine_threadsafe to run the async function in the event loop
                    asyncio.run_coroutine_threadsafe(
                        self._process_leader_data(json_data), 
                        self._livekit_loop
                    )
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error: {e} - Raw data: {decoded_data[:100]}")
                except Exception as e:
                    logging.error(f"Error processing received data: {e}", exc_info=True)
        
        # Connect to the room
        await self.livekit_room.connect(url, token)
        logging.info("Teleop follower connected and listening for leader positions")
    
    async def _process_leader_data(self, message):
        """Process leader data asynchronously
        
        Args:
            message: The already parsed JSON message data
        """
        try:
            # Print message info for debugging
            logging.debug(f"Processing message with keys: {list(message.keys())}")
            
            if "leader_arm_positions" in message:
                leader_positions = message["leader_arm_positions"]
                logging.debug(f"Received leader positions keys: {list(leader_positions.keys())}")
                
                # Process each arm position
                for leader_name, positions in leader_positions.items():
                    # Map leader arm name to follower arm name if needed
                    follower_name = self._map_leader_to_follower_name(leader_name)
                    
                    # Check if we have valid position data
                    if not positions:
                        logging.warning(f"Received empty positions for {leader_name}")
                        continue
                        
                    # Show position data info for debugging
                    if isinstance(positions, list):
                        preview = positions[:3] if len(positions) > 3 else positions
                        logging.debug(f"Processing leader arm {leader_name} -> follower arm {follower_name}, positions (list len={len(positions)}): {preview}...")
                    else:
                        logging.debug(f"Processing leader arm {leader_name} -> follower arm {follower_name}, positions (type={type(positions)}): {positions}")
                    
                    if follower_name in self.follower_arms:
                        try:
                            # Make sure positions is a list or appropriate structure
                            if not isinstance(positions, (list, tuple, np.ndarray)):
                                logging.error(f"Expected positions to be a list/array, got {type(positions)}")
                                continue
                            
                            # Check if the position data has the expected length
                            expected_length = len(self.follower_arms[follower_name].motor_names)
                            if len(positions) != expected_length:
                                logging.error(f"Position length mismatch: got {len(positions)}, expected {expected_length} for arm {follower_name}")
                                continue
                                
                            # Convert list to tensor and send to follower arm
                            goal_pos = torch.tensor(positions, dtype=torch.float32)
                            # Execute actions on the follower arm
                            self.write_goal_position_to_follower(follower_name, goal_pos)
                            logging.debug(f"Applied positions to follower arm: {follower_name}")
                        except Exception as e:
                            logging.error(f"Error applying positions to follower arm {follower_name}: {e}", exc_info=True)
                    else:
                        logging.warning(f"Received positions for unknown arm: {leader_name} -> {follower_name}")
            else:
                logging.warning(f"Received message without leader_arm_positions key: {list(message.keys())}")
        except Exception as e:
            logging.error(f"Unexpected error processing leader arm positions: {e}", exc_info=True)

    def _map_leader_to_follower_name(self, leader_name):
        """Map a leader arm name to the corresponding follower arm name.
        This is needed because the leader and follower might use different naming conventions.
        
        In most cases, the names will be the same, but this allows for custom mapping if needed.
        """
        # Check if we need to map the name (for example, if leader arms have different names than follower arms)
        # For now, just return the same name
        return leader_name

    def write_goal_position_to_follower(self, name, goal_pos):
        """Write goal position to a follower arm.
        
        Args:
            name: Name of the follower arm
            goal_pos: Position to set
            
        Returns:
            The goal position that was actually sent (which may be clamped)
        """
        before_fwrite_t = time.perf_counter()
        
        # Cap goal position when too far away from present position.
        # Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.follower_arms[name].read("Present_Position")
            present_pos = torch.from_numpy(present_pos)
            goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)
        
        # Convert to numpy and write to motor
        goal_pos_np = goal_pos.numpy().astype(np.float32)
        self.follower_arms[name].write("Goal_Position", goal_pos_np)
        
        self.logs[f"write_follower_{name}_goal_pos_dt_s"] = time.perf_counter() - before_fwrite_t
        return goal_pos

    def activate_calibration(self):
        """After calibration all motors function in human interpretable ranges.
        Rotations are expressed in degrees in nominal range of [-180, 180],
        and linear motions (like gripper of Aloha) in nominal range of [0, 100].
        """

        def load_or_run_calibration_(name, arm, arm_type):
            arm_id = get_arm_id(name, arm_type)
            arm_calib_path = self.calibration_dir / f"{arm_id}.json"

            if arm_calib_path.exists():
                with open(arm_calib_path) as f:
                    calibration = json.load(f)
            else:
                # TODO(rcadene): display a warning in __init__ if calibration file not available
                print(f"Missing calibration file '{arm_calib_path}'")

                if self.robot_type in ["koch", "koch_bimanual", "aloha"]:
                    from lerobot.common.robot_devices.robots.dynamixel_calibration import run_arm_calibration

                    calibration = run_arm_calibration(arm, self.robot_type, name, arm_type)

                elif self.robot_type in ["so100", "so101", "moss", "lekiwi", "so100bimanual", "so100bimanual_teleop_follower", "so100bimanual_teleop_leader"]:
                    from lerobot.common.robot_devices.robots.feetech_calibration import (
                        run_arm_manual_calibration,
                    )

                    calibration = run_arm_manual_calibration(arm, self.robot_type, name, arm_type)

                print(f"Calibration is done! Saving calibration file '{arm_calib_path}'")
                arm_calib_path.parent.mkdir(parents=True, exist_ok=True)
                with open(arm_calib_path, "w") as f:
                    json.dump(calibration, f)

            return calibration

        for name, arm in self.follower_arms.items():
            calibration = load_or_run_calibration_(name, arm, "follower")
            arm.set_calibration(calibration)
        for name, arm in self.leader_arms.items():
            calibration = load_or_run_calibration_(name, arm, "leader")
            arm.set_calibration(calibration)

    def set_koch_robot_preset(self):
        def set_operating_mode_(arm):
            from lerobot.common.robot_devices.motors.dynamixel import TorqueMode

            if (arm.read("Torque_Enable") != TorqueMode.DISABLED.value).any():
                raise ValueError("To run set robot preset, the torque must be disabled on all motors.")

            # Use 'extended position mode' for all motors except gripper, because in joint mode the servos can't
            # rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while assembling the arm,
            # you could end up with a servo with a position 0 or 4095 at a crucial point See [
            # https://emanual.robotis.com/docs/en/dxl/x/x_series/#operating-mode11]
            all_motors_except_gripper = [name for name in arm.motor_names if name != "gripper"]
            if len(all_motors_except_gripper) > 0:
                # 4 corresponds to Extended Position on Koch motors
                arm.write("Operating_Mode", 4, all_motors_except_gripper)

            # Use 'position control current based' for gripper to be limited by the limit of the current.
            # For the follower gripper, it means it can grasp an object without forcing too much even tho,
            # it's goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
            # For the leader gripper, it means we can use it as a physical trigger, since we can force with our finger
            # to make it move, and it will move back to its original target position when we release the force.
            # 5 corresponds to Current Controlled Position on Koch gripper motors "xl330-m077, xl330-m288"
            arm.write("Operating_Mode", 5, "gripper")

        for name in self.follower_arms:
            set_operating_mode_(self.follower_arms[name])

            # Set better PID values to close the gap between recorded states and actions
            # TODO(rcadene): Implement an automatic procedure to set optimal PID values for each motor
            self.follower_arms[name].write("Position_P_Gain", 1500, "elbow_flex")
            self.follower_arms[name].write("Position_I_Gain", 0, "elbow_flex")
            self.follower_arms[name].write("Position_D_Gain", 600, "elbow_flex")

        if self.config.gripper_open_degree is not None:
            for name in self.leader_arms:
                set_operating_mode_(self.leader_arms[name])

                # Enable torque on the gripper of the leader arms, and move it to 45 degrees,
                # so that we can use it as a trigger to close the gripper of the follower arms.
                self.leader_arms[name].write("Torque_Enable", 1, "gripper")
                self.leader_arms[name].write("Goal_Position", self.config.gripper_open_degree, "gripper")

    def set_aloha_robot_preset(self):
        def set_shadow_(arm):
            # Set secondary/shadow ID for shoulder and elbow. These joints have two motors.
            # As a result, if only one of them is required to move to a certain position,
            # the other will follow. This is to avoid breaking the motors.
            if "shoulder_shadow" in arm.motor_names:
                shoulder_idx = arm.read("ID", "shoulder")
                arm.write("Secondary_ID", shoulder_idx, "shoulder_shadow")

            if "elbow_shadow" in arm.motor_names:
                elbow_idx = arm.read("ID", "elbow")
                arm.write("Secondary_ID", elbow_idx, "elbow_shadow")

        for name in self.follower_arms:
            set_shadow_(self.follower_arms[name])

        for name in self.leader_arms:
            set_shadow_(self.leader_arms[name])

        for name in self.follower_arms:
            # Set a velocity limit of 131 as advised by Trossen Robotics
            self.follower_arms[name].write("Velocity_Limit", 131)

            # Use 'extended position mode' for all motors except gripper, because in joint mode the servos can't
            # rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while assembling the arm,
            # you could end up with a servo with a position 0 or 4095 at a crucial point See [
            # https://emanual.robotis.com/docs/en/dxl/x/x_series/#operating-mode11]
            all_motors_except_gripper = [
                name for name in self.follower_arms[name].motor_names if name != "gripper"
            ]
            if len(all_motors_except_gripper) > 0:
                # 4 corresponds to Extended Position on Aloha motors
                self.follower_arms[name].write("Operating_Mode", 4, all_motors_except_gripper)

            # Use 'position control current based' for follower gripper to be limited by the limit of the current.
            # It can grasp an object without forcing too much even tho,
            # it's goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
            # 5 corresponds to Current Controlled Position on Aloha gripper follower "xm430-w350"
            self.follower_arms[name].write("Operating_Mode", 5, "gripper")

            # Note: We can't enable torque on the leader gripper since "xc430-w150" doesn't have
            # a Current Controlled Position mode.

        if self.config.gripper_open_degree is not None:
            warnings.warn(
                f"`gripper_open_degree` is set to {self.config.gripper_open_degree}, but None is expected for Aloha instead",
                stacklevel=1,
            )

    def set_so100_robot_preset(self):
        for name in self.follower_arms:
            # Mode=0 for Position Control
            self.follower_arms[name].write("Mode", 0)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            self.follower_arms[name].write("P_Coefficient", 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self.follower_arms[name].write("I_Coefficient", 0)
            self.follower_arms[name].write("D_Coefficient", 32)
            # Close the write lock so that Maximum_Acceleration gets written to EPROM address,
            # which is mandatory for Maximum_Acceleration to take effect after rebooting.
            self.follower_arms[name].write("Lock", 0)
            # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
            # the motors. Note: this configuration is not in the official STS3215 Memory Table
            self.follower_arms[name].write("Maximum_Acceleration", 254)
            self.follower_arms[name].write("Acceleration", 254)

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        # For teleop follower, we don't need to do the regular teleop as commands are received via Livekit
        if self.is_teleop_follower:
            # Just return empty data - the actual position updates happen in the Livekit callback
            if record_data:
                return {}, {}
            return None

        # Prepare to assign the position of the leader to the follower
        leader_pos = {}
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            leader_pos[name] = self.leader_arms[name].read("Present_Position")
            leader_pos[name] = torch.from_numpy(leader_pos[name])
            self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t

        # Livekit publish (if teleop leader)
        if self.is_teleop_leader and self.livekit_room is not None:
            try:
                leader_flat = {k: v.tolist() for k, v in leader_pos.items()}
                data = json.dumps({"leader_arm_positions": leader_flat}).encode("utf-8")
                async def send_packet():
                    await self.livekit_room.local_participant.publish_data(
                        data, reliable=False, topic="leader_arm_positions"
                    )
                    print(f"Publishing leader arm positions to Livekit: {data}")
                if self._livekit_loop:
                    asyncio.run_coroutine_threadsafe(send_packet(), self._livekit_loop)
            except Exception as e:
                logging.error(f"Failed to publish leader arm positions to Livekit: {e}")

        # Send goal position to the follower
        follower_goal_pos = {}
        # Normal local teleop (not follower)
        if not self.is_teleop_follower:
            for name in self.follower_arms:
                before_fwrite_t = time.perf_counter()
                goal_pos = leader_pos[name]
                # Cap goal position when too far away from present position.
                # Slower fps expected due to reading from the follower.
                if self.config.max_relative_target is not None:
                    present_pos = self.follower_arms[name].read("Present_Position")
                    present_pos = torch.from_numpy(present_pos)
                    goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

                # Used when record_data=True
                follower_goal_pos[name] = goal_pos

                goal_pos = goal_pos.numpy().astype(np.float32)
                self.follower_arms[name].write("Goal_Position", goal_pos)
                self.logs[f"write_follower_{name}_goal_pos_dt_s"] = time.perf_counter() - before_fwrite_t

        return {}, {}
    
        # Early exit when recording data is not requested
        if not record_data:
            return None
        
        # TODO(rcadene): Add velocity and other info
        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = self.follower_arms[name].read("Present_Position")
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = torch.cat(state)

        # Create action by concatenating follower goal position
        action = []
        for name in self.follower_arms:
            if name in follower_goal_pos:
                action.append(follower_goal_pos[name])
        action = torch.cat(action)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = self.follower_arms[name].read("Present_Position")
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = torch.cat(state)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """Command the follower arms to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action: tensor containing the concatenated goal positions for the follower arms.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        from_idx = 0
        to_idx = 0
        action_sent = []
        for name in self.follower_arms:
            # Get goal position of each follower arm by splitting the action vector
            to_idx += len(self.follower_arms[name].motor_names)
            goal_pos = action[from_idx:to_idx]
            from_idx = to_idx

            # Cap goal position when too far away from present position.
            # Slower fps expected due to reading from the follower.
            if self.config.max_relative_target is not None:
                present_pos = self.follower_arms[name].read("Present_Position")
                present_pos = torch.from_numpy(present_pos)
                goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

            # Save tensor to concat and return
            action_sent.append(goal_pos)

            # Send goal position to each follower
            goal_pos = goal_pos.numpy().astype(np.float32)
            self.follower_arms[name].write("Goal_Position", goal_pos)

        return torch.cat(action_sent)

    def print_logs(self):
        pass
        # TODO(aliberts): move robot-specific logs logic here

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        for name in self.follower_arms:
            self.follower_arms[name].disconnect()

        for name in self.leader_arms:
            self.leader_arms[name].disconnect()

        for name in self.cameras:
            self.cameras[name].disconnect()

        # Livekit cleanup
        if (self.is_teleop_leader or self.is_teleop_follower) and self.livekit_room is not None and self._livekit_loop is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(self.livekit_room.disconnect(), self._livekit_loop)
                fut.result(timeout=5)
            except Exception as e:
                logging.error(f"Error disconnecting Livekit room: {e}")
            self.livekit_room = None
            self._livekit_loop.call_soon_threadsafe(self._livekit_loop.stop)
            if self._livekit_thread is not None:
                self._livekit_thread.join(timeout=2)
            self._livekit_loop = None
            self._livekit_thread = None
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
