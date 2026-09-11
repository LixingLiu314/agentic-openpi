"""Serve a complete backbone-gradient checkpoint with the existing native Piper client."""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from openpi.policies.backbone_gradient_policy import create_backbone_gradient_policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


class DefaultPromptPolicy:
    def __init__(self, policy, prompt):
        self.policy, self.prompt = policy, prompt

    def infer(self, observation):
        observation = dict(observation)
        observation.setdefault("prompt", self.prompt)
        return self.policy.infer(observation)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--prompt", default="Put the eggplant into the box")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    policy = create_backbone_gradient_policy(args.checkpoint, device=args.device)
    names = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    observation = {"state":np.zeros(14, dtype=np.float32), "prompt":args.prompt,
                   "images":{name:np.zeros((480,640,3), dtype=np.uint8) for name in names}}
    noise = np.random.default_rng(42).standard_normal((50,32), dtype=np.float32)
    outputs = [policy.infer(observation, noise=noise) for _ in range(3)]
    assert all(value["actions"].shape == (50,14) and np.isfinite(value["actions"]).all() for value in outputs)
    np.testing.assert_array_equal(outputs[-2]["actions"], outputs[-1]["actions"])
    metadata = dict(policy.metadata, default_prompt=args.prompt, action_horizon=50, action_dt_s=1/30,
                    camera_names=list(names), state_dim=14, gripper_indices=[6,13], gripper_unit="metres",
                    subtask_input_required=False, reset_pose=None)
    logging.info("Backbone-gradient %s ready at %s:%d: %s", metadata["mode"], args.host, args.port, args.checkpoint)
    WebsocketPolicyServer(DefaultPromptPolicy(policy, args.prompt), host=args.host, port=args.port,
                          metadata=metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
