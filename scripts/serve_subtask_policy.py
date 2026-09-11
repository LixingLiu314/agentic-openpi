"""Serve M3, R1, or complete backbone-gradient weights with the native Piper contract."""

import argparse
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch

from openpi.policies.subtask_policy import create_subtask_policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from piper_checkpoint import checkpoint_info


def load_policy(checkpoint, *, device="cuda:0", parent_checkpoint=None, allow_unqualified_r1=False):
    info = checkpoint_info(checkpoint, parent_checkpoint=parent_checkpoint)
    if info["kind"] == "r1":
        if info["experimental"] and not allow_unqualified_r1:
            raise ValueError("R1 candidate gates did not pass; experimental loading requires --allow-unqualified-r1")
        from openpi.policies.subtask_transition_policy import create_transition_policy
        return create_transition_policy(info["path"], device=device, num_steps=10,
                                        parent_checkpoint=info["parent_checkpoint"],
                                        require_candidate=not allow_unqualified_r1)
    if allow_unqualified_r1:
        raise ValueError("--allow-unqualified-r1 only applies to R1 checkpoints")
    if info["kind"] == "recurrent_subtask":
        if info.get("experiment") in ("decision_prefix", "decision_grounded"):
            from openpi.policies.decision_recurrent_policy import create_decision_recurrent_policy
            return create_decision_recurrent_policy(info["path"], device=device, num_steps=10)
        if info.get("experiment") in ("semantic_s", "semantic_s_actionrank"):
            from openpi.policies.semantic_recurrent_policy import create_semantic_recurrent_policy
            return create_semantic_recurrent_policy(info["path"], device=device, num_steps=10)
        if info.get("label_version") == "reach_arm_v1":
            from openpi.policies.reach_arm_subtask_policy import create_reach_arm_policy
            return create_reach_arm_policy(info["path"], device=device, num_steps=10)
        from openpi.policies.recurrent_subtask_policy import create_recurrent_subtask_policy
        return create_recurrent_subtask_policy(info["path"], device=device, num_steps=10)
    if info["kind"] == "official_backbone_grad":
        from openpi.policies.official_gradient_policy import create_official_gradient_policy
        return create_official_gradient_policy(info["path"], device=device, num_steps=10)
    if info["kind"] == "backbone_grad":
        from openpi.policies.backbone_gradient_policy import create_backbone_gradient_policy
        return create_backbone_gradient_policy(info["path"], device=device, num_steps=10)
    return create_subtask_policy(info["path"], device=device, num_steps=10)


class LoggedPolicy:
    def __init__(self, policy, prompt, log_dir):
        self.policy = policy
        self.prompt = prompt
        self.log_dir = log_dir
        self.index = 0

    def infer(self, observation):
        observation = dict(observation)
        observation.setdefault("prompt", self.prompt)
        result = self.policy.infer(observation)
        self.index += 1
        record = {
            "query": self.index,
            "time": time.time(),
            "subtask": result["subtask"],
            "subtask_status": result["subtask_status"],
            "subtask_score": result["subtask_score"],
            "policy_timing": result["policy_timing"],
            "state": np.asarray(observation["state"]).tolist(),
            "first_action": result["actions"][0].tolist(),
        }
        if self.log_dir is not None:
            with (self.log_dir / "server_queries.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        logging.debug("query=%d subtask=%r status=%s infer_ms=%.1f", self.index,
                     result["subtask"], result["subtask_status"], result["policy_timing"]["infer_ms"])
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, help="R1 M3 base; defaults to initialize_from relocated into this checkout")
    parser.add_argument("--allow-unqualified-r1", action="store_true", help="Load an R1 experiment that has not passed candidate gates; integrity checks remain mandatory")
    parser.add_argument("--device", default="cuda:0", help="Policy device; cpu is available for no-motion loading checks")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="Put the eggplant into the box")
    parser.add_argument("--log-dir", type=Path, help="Optional legacy debug files; omitted for normal runs")
    parser.add_argument("--warmup-observation", type=Path, help="Optional real NPZ snapshot; otherwise synthetic warmup")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.log_dir is not None:
        args.log_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    policy = load_policy(args.checkpoint, device=args.device, parent_checkpoint=args.parent_checkpoint,
                         allow_unqualified_r1=args.allow_unqualified_r1)
    if torch.device(args.device).type == "cpu":
        # CPU checks use float32; BF16 may be emulated on the robot's desktop CPU.
        policy.model.float()
    from openpi.policies.rtc_policy import RTCPolicy
    policy = RTCPolicy(policy)
    names = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    if args.warmup_observation is not None:
        with np.load(args.warmup_observation, allow_pickle=False) as data:
            observation = {"state": data["state"].copy(), "prompt": args.prompt,
                           "images": {name: data[name].copy() for name in names}}
    else:
        observation = {"state": np.zeros(14, dtype=np.float32), "prompt": args.prompt,
                       "images": {name: np.zeros((480, 640, 3), dtype=np.uint8) for name in names}}
    noise = np.random.default_rng(42).standard_normal((50, 32), dtype=np.float32)
    outputs = []
    for index in range(3):
        logging.info("Policy warmup %d/3 on %s", index + 1, args.device)
        warm_policy = policy.new_session() if hasattr(policy, "new_session") else policy
        outputs.append(warm_policy.infer(observation, noise=noise))
    assert all(x["actions"].shape == (50, 14) and np.isfinite(x["actions"]).all() for x in outputs)
    assert np.array_equal(outputs[1]["actions"], outputs[2]["actions"])
    report = {"checkpoint": str(args.checkpoint.resolve()), "prompt": args.prompt,
              "device": args.device, "action_shape": [50, 14],
              "fixed_noise_repeat_equal": True, "subtask": outputs[-1]["subtask"],
              "subtask_status": outputs[-1]["subtask_status"],
              "warmup_infer_ms": [x["policy_timing"]["infer_ms"] for x in outputs],
              "first_action": outputs[-1]["actions"][0].tolist(),
              "max_first_joint_delta": float(np.max(np.abs(
                  (outputs[-1]["actions"][0] - observation["state"])[[0,1,2,3,4,5,7,8,9,10,11,12]]))),
              "scope": "Real snapshot" if args.warmup_observation else "Synthetic warmup; not a task evaluation"}
    if args.log_dir is not None:
        (args.log_dir / "server_warmup.json").write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(args.log_dir / "server_warmup_actions.npz", actions=outputs[-1]["actions"])
    metadata = dict(policy.metadata, default_prompt=args.prompt, action_horizon=50, action_dt_s=1/30,
                    camera_names=["cam_high", "cam_left_wrist", "cam_right_wrist"],
                    state_dim=14, gripper_indices=[6, 13], gripper_unit="metres",
                    subtask_input_required=False, reset_pose=None)
    logging.info("%s ready: %s, port=%d, warm infer=%.0fms",
                 ("Recurrent S / B-" + policy.metadata["mode"]) if policy.metadata.get("stage") == "recurrent_subtask" else ("Official B-" + policy.metadata["mode"]) if policy.metadata.get("variant") == "official_pi05_backbone_v1" else ("Backbone " + policy.metadata["mode"]) if policy.metadata.get("variant") == "action_backbone_v1" else "R1 experiment" if policy.metadata.get("experimental") else (
                     "R1" if policy.metadata.get("variant") == "r1_boundary_ce_v1" else "M3"),
                 args.checkpoint, args.port,
                 outputs[-1]["policy_timing"]["infer_ms"])
    if hasattr(policy, "new_session"):
        from serve_recurrent_subtask_piper import RecurrentWebsocketServer
        RecurrentWebsocketServer(policy, prompt=args.prompt, host=args.host, port=args.port,
                                 metadata=metadata).serve_forever()
        return
    WebsocketPolicyServer(LoggedPolicy(policy, args.prompt, args.log_dir), host=args.host,
                          port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
