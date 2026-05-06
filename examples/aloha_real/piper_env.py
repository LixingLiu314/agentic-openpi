"""
AgiLex Piper environment wrapper for the openpi_client Runtime.

Mirrors examples/aloha_real/env.py but uses piper_real_env.PiperRealEnv
(ROS topic-based, no Interbotix SDK dependency) instead of real_env.RealEnv.
"""

from typing import List, Optional  # noqa: UP035

import einops
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.aloha_real import piper_real_env as _real_env


class PiperAlohaRealEnvironment(_environment.Environment):
    """openpi Runtime environment backed by the AgiLex Piper robot."""

    def __init__(
        self,
        reset_position: Optional[List[float]] = None,  # noqa: UP006,UP007
        render_height: int = 224,
        render_width: int = 224,
        **piper_kwargs,
    ) -> None:
        """
        Args:
            reset_position: 6-DOF reset pose received from policy server metadata.
            render_height / render_width: image resize target (must match training).
            **piper_kwargs: forwarded to PiperRealEnv (topic overrides, gripper
                limits, reset_move_time, etc.).
        """
        self._env = _real_env.make_real_env(
            init_node=True,
            reset_position=reset_position,
            **piper_kwargs,
        )
        self._render_height = render_height
        self._render_width  = render_width
        self._ts = None

    @override
    def reset(self) -> None:
        self._ts = self._env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts.observation

        for cam_name in list(obs["images"].keys()):
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(
                    obs["images"][cam_name], self._render_height, self._render_width
                )
            )
            # Policy server expects CHW format.
            obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")

        return {
            "state":  obs["qpos"],
            "images": obs["images"],
        }

    @override
    def apply_action(self, action: dict) -> None:
        self._ts = self._env.step(action["actions"])
