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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig, RemoteRobotConfig


@RobotConfig.register_subclass("tri_so100_follower")
@dataclass
class TriSO100FollowerConfig(RobotConfig):
    arm1_port: str
    arm2_port: str
    arm3_port: str

    # Optional
    arm1_disable_torque_on_disconnect: bool = True
    arm1_max_relative_target: float | dict[str, float] | None = None
    arm1_use_degrees: bool = False

    arm2_disable_torque_on_disconnect: bool = True
    arm2_max_relative_target: float | dict[str, float] | None = None
    arm2_use_degrees: bool = False

    arm3_disable_torque_on_disconnect: bool = True
    arm3_max_relative_target: float | dict[str, float] | None = None
    arm3_use_degrees: bool = False

    # cameras (shared between all arms)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@RemoteRobotConfig.register_subclass("tri_so100_remote_follower")
@dataclass
class TriSO100RemoteFollowerConfig(RemoteRobotConfig):
    disable_torque_on_disconnect: bool = True
    max_relative_target: int | None = None
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    use_degrees: bool = False


