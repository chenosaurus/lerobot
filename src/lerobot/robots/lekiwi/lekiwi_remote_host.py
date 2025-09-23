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
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.transport.livekit_service import LiveKitService, LiveKitServiceHandler

from .config_lekiwi import LeKiwiConfig
from .lekiwi import LeKiwi

try:
    from livekit import rtc

    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False

try:
    from dotenv import load_dotenv  # type: ignore

    _DOTENV_AVAILABLE = True
except Exception:  # pragma: no cover
    _DOTENV_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LeKiwiRemoteHostConfig:
    """Runtime configuration for the LiveKit host loop (not user-exposed schema)."""

    # Duration of the application
    connection_time_s: int = 3000
    # Watchdog: stop the robot if no command is received for over 0.5 seconds.
    watchdog_timeout_ms: int = 500
    # Max host loop frequency
    max_loop_freq_hz: int = 30


class _LeKiwiRemoteHostHandler(LiveKitServiceHandler):
    def __init__(self, host: "LeKiwiRemoteHost"):
        self.host = host

    def on_data_received(self, data: "rtc.DataPacket") -> None:  # type: ignore[name-defined]
        try:
            if data.topic == "teleop_action":
                action_data = json.loads(data.data.decode("utf-8"))
                self.host._handle_action_message(action_data)
        except Exception as e:
            logger.error(f"Error processing data packet: {e}")

    def on_connected(self) -> None:
        try:
            self.host._on_livekit_connected()
        except Exception as e:
            logger.error(f"Error handling LiveKit on_connected: {e}")


class LeKiwiRemoteHost:
    """
    LiveKit transport host that bridges a local LeKiwi robot to a remote client.

    - Receives actions over topic 'teleop_action'
    - Sends scalar observations over topic 'teleop_observation'
    - Publishes camera frames as LiveKit video tracks
    """

    def __init__(self, livekit_url: str, livekit_token: str, robot_config: LeKiwiConfig,
                 runtime_cfg: LeKiwiRemoteHostConfig | None = None) -> None:
        if not LIVEKIT_AVAILABLE:
            raise ImportError("LiveKit SDK is required. Install with: pip install livekit")

        # Load .env so LIVEKIT_URL/LIVEKIT_TOKEN can be read when main() is not used
        if _DOTENV_AVAILABLE:
            with contextlib.suppress(Exception):
                load_dotenv()

        # Fallback from environment variables if blank strings are provided
        livekit_url = livekit_url or os.environ.get("LIVEKIT_URL", "")
        livekit_token = livekit_token or os.environ.get("LIVEKIT_TOKEN", "")
        if not livekit_url or not livekit_token:
            raise RuntimeError("LIVEKIT_URL and LIVEKIT_TOKEN must be set (via args or .env/env)")

        self.robot = LeKiwi(robot_config)
        self.robot.connect()

        self.runtime_cfg = runtime_cfg or LeKiwiRemoteHostConfig()

        self._handler = _LeKiwiRemoteHostHandler(self)
        self._livekit = LiveKitService(livekit_url, livekit_token, self._handler)

        # Action cache updated from data channel
        self._latest_action: dict[str, Any] | None = None

        # Video publishing state
        self._video_sources: dict[str, "rtc.VideoSource"] = {}  # type: ignore[name-defined]
        self._video_tracks: dict[str, "rtc.LocalVideoTrack"] = {}  # type: ignore[name-defined]
        self._published_tracks: set[str] = set()

    # ----- LiveKit control plane -----
    def connect(self) -> None:
        self._livekit.connect(timeout=10.0)

    def disconnect(self) -> None:
        with contextlib.suppress(RuntimeError):
            self._livekit.disconnect()
        # Cleanup robot
        with contextlib.suppress(Exception):
            self.robot.disconnect()

    def _handle_action_message(self, action_data: dict[str, Any]) -> None:
        self._latest_action = action_data

    # ----- Video publishing -----
    def _ensure_video_track(self, camera_name: str, width: int, height: int) -> None:
        if camera_name not in self._video_tracks:
            source = rtc.VideoSource(width, height)
            track = rtc.LocalVideoTrack.create_video_track(f"camera_{camera_name}", source)
            self._video_sources[camera_name] = source
            self._video_tracks[camera_name] = track
            logger.info(f"Created video track '{camera_name}'")

        # Try publishing now (safe to call repeatedly)
        self._maybe_publish_track(camera_name)

    def _maybe_publish_track(self, camera_name: str) -> None:
        logger.info(f"Maybe publish track '{camera_name}'")
        if (
            camera_name in self._video_tracks
            and camera_name not in self._published_tracks
            and self._livekit.room
            and self._livekit.room.local_participant
            and self._livekit._event_loop
        ):
            track = self._video_tracks[camera_name]
            options = rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_CAMERA,
                simulcast=True,
                video_encoding=rtc.VideoEncoding(max_framerate=30, max_bitrate=2_000_000),
                video_codec=rtc.VideoCodec.H264,
            )
            future = asyncio.run_coroutine_threadsafe(
                self._livekit.room.local_participant.publish_track(track, options),
                self._livekit._event_loop,
            )

            def _done(f):
                try:
                    _ = f.result()
                    self._published_tracks.add(camera_name)
                    logger.info(f"Published video track '{camera_name}'")
                except Exception as e:
                    logger.error(f"Failed to publish track '{camera_name}': {e}")

            future.add_done_callback(_done)

    def _on_livekit_connected(self) -> None:
        for cam_name in list(self._video_tracks.keys()):
            self._maybe_publish_track(cam_name)

    def _publish_frame(self, camera_name: str, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        h, w = frame.shape[:2]
        self._ensure_video_track(camera_name, w, h)

        if frame.shape[2] == 3:
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = frame
            rgba[:, :, 3] = 255
            data = rgba.flatten().tobytes()
            buf_type = rtc.VideoBufferType.RGBA
        elif frame.shape[2] == 4:
            data = frame.flatten().tobytes()
            buf_type = rtc.VideoBufferType.RGBA
        else:
            logger.warning(f"Unsupported channels for camera {camera_name}: {frame.shape}")
            return

        vf = rtc.VideoFrame(w, h, buf_type, data)
        self._video_sources[camera_name].capture_frame(vf)

    # ----- Main loop -----
    def run(self) -> None:
        """Run the host loop: receive actions, command robot, publish obs and video."""
        self.connect()

        try:
            last_cmd_time = time.time()
            watchdog_active = False

            start = time.perf_counter()
            duration = 0.0
            while duration < self.runtime_cfg.connection_time_s:
                loop_start = time.time()

                # Apply latest action if any
                action = self._latest_action
                if action is not None:
                    try:
                        _ = self.robot.send_action(action)
                        last_cmd_time = time.time()
                        watchdog_active = False
                    except Exception as e:
                        logger.error(f"Failed to apply action: {e}")

                now = time.time()
                if (now - last_cmd_time > self.runtime_cfg.watchdog_timeout_ms / 1000) and not watchdog_active:
                    logger.warning(
                        f"No command for > {self.runtime_cfg.watchdog_timeout_ms} ms. Stopping the base."
                    )
                    watchdog_active = True
                    with contextlib.suppress(Exception):
                        self.robot.stop_base()

                # Gather observation
                obs = self.robot.get_observation()

                # Split into scalar + camera frames
                scalar_obs: dict[str, Any] = {}
                for key, value in obs.items():
                    if isinstance(value, np.ndarray) and value.ndim == 3:
                        # Publish video
                        self._publish_frame(key, value)
                    else:
                        # Publish only values that match the action schema (kept simple)
                        if isinstance(value, (int, float)):
                            scalar_obs[key] = float(value)

                # Publish scalar observation
                if scalar_obs:
                    try:
                        self._livekit.publish_json_sync(scalar_obs, "teleop_observation", reliable=False, timeout=0.5)
                    except Exception as e:
                        logger.warning(f"Failed to publish observation: {e}")

                # pacing
                elapsed = time.time() - loop_start
                time.sleep(max(1 / self.runtime_cfg.max_loop_freq_hz - elapsed, 0))
                duration = time.perf_counter() - start

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.disconnect()


def main() -> None:
    """Example CLI usage for manual runs.

    This function assumes environment-provided URL/TOKEN and a default LeKiwiConfig.
    """
    import os

    logging.basicConfig(level=logging.INFO)

    if _DOTENV_AVAILABLE:
        env_path = "/home/dc/workspace/lerobot/.env"
        load_dotenv(dotenv_path=env_path)

    livekit_url = os.environ.get("LIVEKIT_URL", "")
    livekit_token = os.environ.get("LIVEKIT_FOLLOWER_TOKEN", "")
    if not livekit_url or not livekit_token:
        raise RuntimeError("LIVEKIT_URL and LIVEKIT_FOLLOWER_TOKEN must be set in the environment.")

    host = LeKiwiRemoteHost(livekit_url, livekit_token, LeKiwiConfig())
    host.run()


if __name__ == "__main__":
    main()


