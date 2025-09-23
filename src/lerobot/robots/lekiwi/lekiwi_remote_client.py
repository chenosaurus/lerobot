#!/usr/bin/env python

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

import asyncio
import contextlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.robots.config import RemoteRobotConfig, RobotConfig
from lerobot.robots.robot import Robot
from lerobot.transport.livekit_service import LiveKitService, LiveKitServiceHandler

try:
    from livekit import rtc

    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False

try:
    # Optional dependency; if not present, we skip silent fallback
    from dotenv import load_dotenv  # type: ignore

    _DOTENV_AVAILABLE = True
except Exception:  # pragma: no cover - best-effort optional import
    _DOTENV_AVAILABLE = False

logger = logging.getLogger(__name__)


@RobotConfig.register_subclass("lekiwi_remote_client")
@dataclass
class LeKiwiRemoteClientConfig(RemoteRobotConfig):
    """
    LiveKit-backed remote client for LeKiwi.

    Uses LiveKit for both control (data channels) and video (WebRTC tracks).
    """


class _LeKiwiRemoteClientHandler(LiveKitServiceHandler):
    """LiveKit event handler for the LeKiwi remote client."""

    def __init__(self, client: "LeKiwiRemoteClient"):
        self.client = client

    def on_data_received(self, data: "rtc.DataPacket") -> None:  # type: ignore[name-defined]
        try:
            if data.topic == "teleop_observation":
                observation_data = json.loads(data.data.decode("utf-8"))
                self.client._handle_observation_message(observation_data)
        except Exception as e:
            logger.error(f"Error processing data packet: {e}")


class LeKiwiRemoteClient(Robot):
    """
    Remote client that communicates with a LeKiwi host via LiveKit.

    - Sends actions on topic 'teleop_action'
    - Receives observations on topic 'teleop_observation'
    - Receives video via subscribed tracks and exposes frames in observation
    """

    config_class = LeKiwiRemoteClientConfig
    name = "lekiwi_remote_client"

    def __init__(self, config: LeKiwiRemoteClientConfig):
        super().__init__(config)

        if not LIVEKIT_AVAILABLE:
            raise ImportError("LiveKit SDK is required. Install with: pip install livekit")

        # Best-effort: load .env so environment variables are available
        if _DOTENV_AVAILABLE:
            try:
                load_dotenv()
            except Exception:  # pragma: no cover - ignore dotenv failures
                pass

        self.config = config

        # Allow fallback to environment variables if config fields are empty
        if not getattr(self.config, "livekit_url", None):
            env_url = os.environ.get("LIVEKIT_URL")
            if env_url:
                setattr(self.config, "livekit_url", env_url)
        if not getattr(self.config, "livekit_token", None):
            env_token = os.environ.get("LIVEKIT_TOKEN")
            if env_token:
                setattr(self.config, "livekit_token", env_token)

        # LiveKit service
        self._handler = _LeKiwiRemoteClientHandler(self)
        self._livekit_service = LiveKitService(config.livekit_url, config.livekit_token, self._handler)

        # State caches
        self._last_observation: dict[str, Any] = {}
        self._last_action: dict[str, Any] = {}
        self._state_lock = threading.Lock()

        # Video subscription handling
        self._video_streams: dict[str, "rtc.VideoStream"] = {}  # type: ignore[name-defined]
        self._video_tasks: dict[str, asyncio.Task] = {}
        self._video_lock = threading.Lock()

    # ----- Feature definitions -----
    @cached_property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                "arm_shoulder_pan.pos",
                "arm_shoulder_lift.pos",
                "arm_elbow_flex.pos",
                "arm_wrist_flex.pos",
                "arm_wrist_roll.pos",
                "arm_gripper.pos",
                "x.vel",
                "y.vel",
                "theta.vel",
            ),
            float,
        )

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        # Only the scalar state is declared statically; video frames are dynamic.
        return self._state_ft

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    # ----- Connection management -----
    @property
    def is_connected(self) -> bool:
        return self._livekit_service.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            self._livekit_service.connect(timeout=10.0)
            self._setup_track_subscriptions()
            logger.info(f"{self} connected to LiveKit server")
        except ConnectionError as e:  # pragma: no cover - pass-through
            raise ConnectionError(f"Failed to connect {self}: {e}") from e

        # No hardware-side calibration required for the remote client.
        # Keep API parity with Robot interface.
        if calibrate:
            try:
                self.calibrate()
            except Exception:
                pass

        # Allow subclasses to configure runtime parameters if needed
        try:
            self.configure()
        except Exception:
            pass

    @property
    def is_calibrated(self) -> bool:
        # Remote client has nothing to calibrate
        return True

    def calibrate(self) -> None:
        # No-op for remote client
        pass

    def configure(self) -> None:
        # No-op for remote client
        pass

    def _setup_track_subscriptions(self) -> None:
        if not self._livekit_service.room:
            return

        @self._livekit_service.room.on("track_subscribed")
        def on_track_subscribed(
            track: "rtc.Track",  # type: ignore[name-defined]
            publication: "rtc.RemoteTrackPublication",  # type: ignore[name-defined]
            participant: "rtc.RemoteParticipant",  # type: ignore[name-defined]
        ):
            # Subscribe to any video track regardless of participant identity
            try:
                from livekit import rtc as _rtc  # local import for typing

                if track.kind == _rtc.TrackKind.KIND_VIDEO:
                    asyncio.create_task(self._handle_video_track(track, participant))
            except Exception as e:  # pragma: no cover - logging only
                logger.error(f"Error subscribing to track: {e}")

        # Also handle already-present participants/tracks
        for participant in self._livekit_service.room.remote_participants.values():
            for publication in participant.track_publications.values():
                if getattr(publication, "track", None) is not None:
                    track = publication.track
                    try:
                        from livekit import rtc as _rtc

                        if track.kind == _rtc.TrackKind.KIND_VIDEO and self._livekit_service._event_loop:
                            asyncio.run_coroutine_threadsafe(
                                self._handle_video_track(track, participant),
                                self._livekit_service._event_loop,
                            )
                    except Exception as e:  # pragma: no cover - logging only
                        logger.error(f"Error handling existing track: {e}")

    async def _handle_video_track(self, track: "rtc.Track", participant: "rtc.RemoteParticipant") -> None:  # type: ignore[name-defined]
        try:
            track_name = getattr(track, "name", None) or f"camera_{track.sid}"
            video_stream = rtc.VideoStream(track)

            with self._video_lock:
                self._video_streams[track_name] = video_stream
                self._video_tasks[track_name] = asyncio.create_task(
                    self._process_video_stream(track_name, video_stream)
                )

            logger.info(f"Started video stream for track: {track_name}")
        except Exception as e:  # pragma: no cover - logging only
            logger.error(f"Error handling video track from {participant.identity}: {e}")

    async def _process_video_stream(self, track_name: str, video_stream: "rtc.VideoStream") -> None:  # type: ignore[name-defined]
        try:
            async for event in video_stream:
                video_frame = event.frame
                frame_array = self._video_frame_to_numpy(video_frame)
                if frame_array is not None:
                    with self._state_lock:
                        self._last_observation[track_name] = frame_array
        except Exception as e:  # pragma: no cover - logging only
            logger.error(f"Error processing video stream {track_name}: {e}")
        finally:
            with self._video_lock:
                self._video_streams.pop(track_name, None)
                self._video_tasks.pop(track_name, None)
            await video_stream.aclose()

    def _video_frame_to_numpy(self, video_frame: "rtc.VideoFrame") -> np.ndarray | None:  # type: ignore[name-defined]
        try:
            rgb_frame = video_frame.convert(rtc.VideoBufferType.RGB24)
            width = rgb_frame.width
            height = rgb_frame.height
            frame_data = np.frombuffer(rgb_frame.data, dtype=np.uint8)
            frame_array = frame_data.reshape((height, width, 3))
            return frame_array
        except Exception as e:  # pragma: no cover - logging only
            logger.error(f"Error converting video frame to numpy: {e}")
            return None

    # ----- Observation / Action API -----
    def _handle_observation_message(self, observation_data: dict[str, Any]) -> None:
        try:
            with self._state_lock:
                self._last_observation.update(observation_data)
        except Exception as e:  # pragma: no cover - logging only
            logger.error(f"Invalid observation message received: {e}")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        with self._state_lock:
            return self._last_observation.copy()

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        try:
            with self._state_lock:
                self._last_action = action.copy()

            try:
                self._livekit_service.publish_json_sync(
                    action,
                    "teleop_action",
                    reliable=False,
                    timeout=1.0,
                )
            except Exception as e:  # pragma: no cover - logging only
                logger.error(f"Error publishing action to LiveKit: {e}")

            return action
        except Exception as e:  # pragma: no cover - logging only
            logger.error(f"Failed to send action: {e}")
            return action

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        logger.info(f"Disconnecting {self} from LiveKit server")

        # Stop video tasks and clear frames
        with self._video_lock:
            for task in self._video_tasks.values():
                if hasattr(task, "cancel"):
                    task.cancel()
            self._video_tasks.clear()
            self._video_streams.clear()

        # Disconnect from LiveKit
        with contextlib.suppress(RuntimeError):
            self._livekit_service.disconnect()

        # Clear state
        self._last_observation = {}
        self._last_action = {}

        logger.info(f"{self} disconnected from LiveKit server")

