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

from dataclasses import dataclass

from ..config import TeleoperatorConfig, RemoteTeleoperatorConfig


@TeleoperatorConfig.register_subclass("tri_so100_leader")
@dataclass
class TriSO100LeaderConfig(TeleoperatorConfig):
    arm1_port: str
    arm2_port: str
    arm3_port: str


@RemoteTeleoperatorConfig.register_subclass("tri_so100_remote_leader")
@dataclass
class TriSO100RemoteLeaderConfig(RemoteTeleoperatorConfig):
    pass


