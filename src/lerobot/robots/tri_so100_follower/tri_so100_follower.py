#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.so100_follower import SO100Follower
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig

from ..robot import Robot
from .config_tri_so100_follower import TriSO100FollowerConfig

logger = logging.getLogger(__name__)


class TriSO100Follower(Robot):
    """
    [Tri-manual SO-100 Follower Arms](https://github.com/TheRobotStudio/SO-ARM100) designed by TheRobotStudio
    This robot can also be easily adapted to use SO-101 follower arms by
    replacing the `SO100Follower` class and `SO100FollowerConfig` accordingly.
    """

    config_class = TriSO100FollowerConfig
    name = "tri_so100_follower"

    def __init__(self, config: TriSO100FollowerConfig):
        super().__init__(config)
        self.config = config

        arm1_config = SO100FollowerConfig(
            id=f"{config.id}_arm1" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.arm1_port,
            disable_torque_on_disconnect=config.arm1_disable_torque_on_disconnect,
            max_relative_target=config.arm1_max_relative_target,
            use_degrees=config.arm1_use_degrees,
            cameras={},
        )

        arm2_config = SO100FollowerConfig(
            id=f"{config.id}_arm2" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.arm2_port,
            disable_torque_on_disconnect=config.arm2_disable_torque_on_disconnect,
            max_relative_target=config.arm2_max_relative_target,
            use_degrees=config.arm2_use_degrees,
            cameras={},
        )

        arm3_config = SO100FollowerConfig(
            id=f"{config.id}_arm3" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.arm3_port,
            disable_torque_on_disconnect=config.arm3_disable_torque_on_disconnect,
            max_relative_target=config.arm3_max_relative_target,
            use_degrees=config.arm3_use_degrees,
            cameras={},
        )

        self.arm1 = SO100Follower(arm1_config)
        self.arm2 = SO100Follower(arm2_config)
        self.arm3 = SO100Follower(arm3_config)
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return (
            {f"arm1_{motor}.pos": float for motor in self.arm1.bus.motors}
            | {f"arm2_{motor}.pos": float for motor in self.arm2.bus.motors}
            | {f"arm3_{motor}.pos": float for motor in self.arm3.bus.motors}
        )

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.arm1.bus.is_connected
            and self.arm2.bus.is_connected
            and self.arm3.bus.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        self.arm1.connect(calibrate)
        self.arm2.connect(calibrate)
        self.arm3.connect(calibrate)

        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        return self.arm1.is_calibrated and self.arm2.is_calibrated and self.arm3.is_calibrated

    def calibrate(self) -> None:
        self.arm1.calibrate()
        self.arm2.calibrate()
        self.arm3.calibrate()

    def configure(self) -> None:
        self.arm1.configure()
        self.arm2.configure()
        self.arm3.configure()

    def setup_motors(self) -> None:
        self.arm1.setup_motors()
        self.arm2.setup_motors()
        self.arm3.setup_motors()

    def get_observation(self) -> dict[str, Any]:
        obs_dict: dict[str, Any] = {}

        # Add arm1 prefix
        arm1_obs = self.arm1.get_observation()
        obs_dict.update({f"arm1_{key}": value for key, value in arm1_obs.items()})

        # Add arm2 prefix
        arm2_obs = self.arm2.get_observation()
        obs_dict.update({f"arm2_{key}": value for key, value in arm2_obs.items()})

        # Add arm3 prefix
        arm3_obs = self.arm3.get_observation()
        obs_dict.update({f"arm3_{key}": value for key, value in arm3_obs.items()})

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Remove prefixes
        arm1_action = {key.removeprefix("arm1_"): value for key, value in action.items() if key.startswith("arm1_")}
        arm2_action = {key.removeprefix("arm2_"): value for key, value in action.items() if key.startswith("arm2_")}
        arm3_action = {key.removeprefix("arm3_"): value for key, value in action.items() if key.startswith("arm3_")}

        send_action_arm1 = self.arm1.send_action(arm1_action)
        send_action_arm2 = self.arm2.send_action(arm2_action)
        send_action_arm3 = self.arm3.send_action(arm3_action)

        # Add prefixes back
        prefixed_send_action_arm1 = {f"arm1_{key}": value for key, value in send_action_arm1.items()}
        prefixed_send_action_arm2 = {f"arm2_{key}": value for key, value in send_action_arm2.items()}
        prefixed_send_action_arm3 = {f"arm3_{key}": value for key, value in send_action_arm3.items()}

        return {**prefixed_send_action_arm1, **prefixed_send_action_arm2, **prefixed_send_action_arm3}

    def disconnect(self):
        self.arm1.disconnect()
        self.arm2.disconnect()
        self.arm3.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()


