"""Native websocket service, one observation memory per connection."""
import argparse
import asyncio
import logging
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import websockets
import websockets.frames
from openpi_client import msgpack_numpy

from openpi.policies.recurrent_subtask_policy import create_recurrent_subtask_policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


class RecurrentWebsocketServer(WebsocketPolicyServer):
    def __init__(self, policy, *, prompt, **kwargs):
        super().__init__(policy, **kwargs)
        self.prompt = prompt

    async def _handler(self, websocket):
        # Metadata probes and independent clients cannot reset another client.
        policy = self._policy.new_session()
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        previous = None
        try:
            while True:
                started = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                obs = dict(obs)
                obs.setdefault("prompt", self.prompt)
                infer_started = time.monotonic()
                result = policy.infer(obs)
                result["server_timing"] = {"infer_ms": (time.monotonic() - infer_started) * 1000}
                if previous is not None:
                    result["server_timing"]["prev_total_ms"] = previous * 1000
                await websocket.send(packer.pack(result))
                previous = time.monotonic() - started
        except websockets.ConnectionClosed:
            pass
        except Exception:
            await websocket.send(traceback.format_exc())
            await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="Inference failed")
            raise
        finally:
            policy.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="Put the eggplant into the box")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    policy = create_recurrent_subtask_policy(args.checkpoint, device=args.device)
    names = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    obs = {"state": np.zeros(14, dtype=np.float32), "prompt": args.prompt,
           "images": {name: np.zeros((480, 640, 3), dtype=np.uint8) for name in names}}
    noise = np.random.default_rng(42).standard_normal((50, 32), dtype=np.float32)
    outputs = []
    for _ in range(2):
        session = policy.new_session()
        outputs.append(session.infer(obs, noise=noise))
    np.testing.assert_array_equal(outputs[0]["actions"], outputs[1]["actions"])
    metadata = dict(policy.metadata, default_prompt=args.prompt, action_horizon=50, action_dt_s=1 / 30,
                    camera_names=list(names), state_dim=14, gripper_indices=[6, 13], gripper_unit="metres",
                    subtask_input_required=False, reset_pose=None)
    RecurrentWebsocketServer(policy, prompt=args.prompt, host=args.host, port=args.port,
                             metadata=metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
