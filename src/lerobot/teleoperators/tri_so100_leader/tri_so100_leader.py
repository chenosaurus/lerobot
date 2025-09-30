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

from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader

from ..teleoperator import Teleoperator
from .config_tri_so100_leader import TriSO100LeaderConfig

logger = logging.getLogger(__name__)


class TriSO100Leader(Teleoperator):
    """
    Tri-manual SO-100 Leader Arms.
    """

    config_class = TriSO100LeaderConfig
    name = "tri_so100_leader"

    def __init__(self, config: TriSO100LeaderConfig):
        super().__init__(config)
        self.config = config

        arm1_config = SO100LeaderConfig(
            id=f"{config.id}1" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.arm1_port,
        )

        arm2_config = SO100LeaderConfig(
            id=f"{config.id}2" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.arm2_port,
        )

        arm3_config = SO100LeaderConfig(
            id=f"{config.id}3" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.arm3_port,
        )

        self.arm1 = SO100Leader(arm1_config)
        self.arm2 = SO100Leader(arm2_config)
        self.arm3 = SO100Leader(arm3_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return (
            {f"arm1_{motor}.pos": float for motor in self.arm1.bus.motors}
            | {f"arm2_{motor}.pos": float for motor in self.arm2.bus.motors}
            | {f"arm3_{motor}.pos": float for motor in self.arm3.bus.motors}
        )

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.arm1.is_connected and self.arm2.is_connected and self.arm3.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.arm1.connect(calibrate)
        self.arm2.connect(calibrate)
        self.arm3.connect(calibrate)

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

    def get_action(self) -> dict[str, float]:
        action_dict: dict[str, float] = {}

        arm1_action = self.arm1.get_action()
        action_dict.update({f"arm1_{key}": value for key, value in arm1_action.items()})

        arm2_action = self.arm2.get_action()
        action_dict.update({f"arm2_{key}": value for key, value in arm2_action.items()})

        arm3_action = self.arm3.get_action()
        action_dict.update({f"arm3_{key}": value for key, value in arm3_action.items()})

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        arm1_feedback = {key.removeprefix("arm1_"): value for key, value in feedback.items() if key.startswith("arm1_")}
        arm2_feedback = {key.removeprefix("arm2_"): value for key, value in feedback.items() if key.startswith("arm2_")}
        arm3_feedback = {key.removeprefix("arm3_"): value for key, value in feedback.items() if key.startswith("arm3_")}

        if arm1_feedback:
            self.arm1.send_feedback(arm1_feedback)
        if arm2_feedback:
            self.arm2.send_feedback(arm2_feedback)
        if arm3_feedback:
            self.arm3.send_feedback(arm3_feedback)

    def disconnect(self) -> None:
        self.arm1.disconnect()
        self.arm2.disconnect()
        self.arm3.disconnect()


