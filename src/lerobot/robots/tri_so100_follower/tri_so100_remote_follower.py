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
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs

from ..remote_robot import RemoteRobot
from .config_tri_so100_follower import TriSO100RemoteFollowerConfig

logger = logging.getLogger(__name__)


class TriSO100RemoteFollower(RemoteRobot):
    """
    Remote Tri-manual SO-100 Followers via WebRTC.

    Receives observations over LiveKit data channel and publishes actions.
    """

    config_class = TriSO100RemoteFollowerConfig
    name = "tri_so100_remote_follower"

    def __init__(self, config: TriSO100RemoteFollowerConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        # Three SO-100 followers, motor names mirror local follower
        joints = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]
        features: dict[str, type] = {}
        for prefix in ("arm1_", "arm2_", "arm3_"):
            for j in joints:
                features[f"{prefix}{j}"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        # If using local cameras on the remote side, expose image shapes
        return {}
        # return {
        #     cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
        #     for cam in self.cameras
        # }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        cameras_connected = all(cam.is_connected for cam in self.cameras.values()) if self.cameras else True
        return super().is_connected and cameras_connected

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate=calibrate)
        for cam in self.cameras.values():
            cam.connect()
        logger.info(f"{self} connected with {len(self.cameras)} cameras.")

    def configure(self) -> None:
        pass

    def setup_motors(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            from lerobot.errors import DeviceNotConnectedError

            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = super().get_observation()

        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        return super().send_action(action)

    def disconnect(self) -> None:
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()
        super().disconnect()


