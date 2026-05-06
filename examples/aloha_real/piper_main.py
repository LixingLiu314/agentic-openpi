"""
Evaluation entry point for AgiLex Piper + openpi policy server.

Mirrors examples/aloha_real/main.py but uses PiperAlohaRealEnvironment
(ROS topic-based) instead of AlohaRealEnvironment (Interbotix SDK).

Usage (three terminals):

  Terminal 1 — robot stack:
    bash examples/Aloha/eval_files/start_robot_stack.sh

  Terminal 2 — policy server (GPU machine):
    uv run scripts/serve_policy.py \\
        --env ALOHA \\
        --default_prompt "pick up the banana" \\
        policy:checkpoint \\
        --policy.config pi05_aloha_banana \\
        --policy.dir checkpoints/pi05_aloha_banana/banana_baseline/5000

  Terminal 3 — this client (robot machine):
    python -m examples.aloha_real.piper_main \\
        --host <server_ip> --port 8000 \\
        --action_horizon 25 --num_episodes 3

  Dry-run (no arm motion, useful for first-time pipeline check):
    python -m examples.aloha_real.piper_main --dry_run
"""

import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.aloha_real import piper_env as _env


@dataclasses.dataclass
class Args:
    # Policy server connection
    host: str = "0.0.0.0"
    port: int = 8000

    # Action chunking: number of steps to execute per model query
    action_horizon: int = 25

    # Episode control
    num_episodes: int = 1
    max_episode_steps: int = 1000

    # Safety: skip publishing joint commands (still runs observation + model)
    dry_run: bool = False

    # PiperRealEnv overrides (leave empty to use defaults from piper_real_env.py)
    img_front_topic:   str = "/camera_f/color/image_raw"
    img_left_topic:    str = "/camera_l/color/image_raw"
    img_right_topic:   str = "/camera_r/color/image_raw"
    joint_left_topic:  str = "/puppet/joint_left"
    joint_right_topic: str = "/puppet/joint_right"
    cmd_left_topic:    str = "/master/joint_left"
    cmd_right_topic:   str = "/master/joint_right"

    # Gripper radian limits — tune to match your puppet hardware
    gripper_open:  float = 4.0
    gripper_close: float = 0.0

    # Seconds to interpolate from current pose to reset pose
    reset_move_time: float = 2.0


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info("Server metadata: %s", ws_client_policy.get_server_metadata())

    metadata = ws_client_policy.get_server_metadata()

    if args.dry_run:
        logging.warning(
            "DRY RUN: observations and model inference are active, "
            "but joint commands will NOT be published to the robot."
        )

    piper_kwargs = dict(
        img_front_topic=args.img_front_topic,
        img_left_topic=args.img_left_topic,
        img_right_topic=args.img_right_topic,
        joint_left_topic=args.joint_left_topic,
        joint_right_topic=args.joint_right_topic,
        cmd_left_topic=args.cmd_left_topic,
        cmd_right_topic=args.cmd_right_topic,
        gripper_open=args.gripper_open,
        gripper_close=args.gripper_close,
        reset_move_time=args.reset_move_time,
        dry_run=args.dry_run,
    )

    environment = _env.PiperAlohaRealEnvironment(
        reset_position=metadata.get("reset_pose"),
        **piper_kwargs,
    )

    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[],
        max_hz=50,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
