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
from typing import Any

from ..remote_teleoperator import RemoteTeleoperator
from .config_bi_so100_leader import BiSO100RemoteLeaderConfig

logger = logging.getLogger(__name__)


class BiSO100RemoteLeader(RemoteTeleoperator):
    """
    Remote Bimanual SO-100 Leader via WebRTC.

    Receives leader actions over LiveKit data channel for two arms.
    """

    config_class = BiSO100RemoteLeaderConfig
    name = "bi_so100_remote_leader"

    def __init__(self, config: BiSO100RemoteLeaderConfig):
        super().__init__(config)
        self.config = config

    @property
    def action_features(self) -> dict[str, type]:
        joints = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]
        features: dict[str, type] = {}
        for prefix in ("left_", "right_"):
            for j in joints:
                features[f"{prefix}{j}"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate=calibrate)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def setup_motors(self) -> None:
        pass

    def get_action(self) -> dict[str, Any]:
        return super().get_action()

    def send_feedback(self, feedback: dict[str, float]) -> None:
        pass

    def disconnect(self) -> None:
        super().disconnect()


