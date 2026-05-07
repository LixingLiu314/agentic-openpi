# -*- coding: utf-8 -*-
"""
Foreact Subgoal Image Prediction Server

WebSocket server that wraps VisualForesightPipeline and serves subgoal image predictions.
Compatible with starVLA's websocket protocol (msgpack_numpy serialization).

Usage:
    CUDA_VISIBLE_DEVICES=0 python server_foreact.py \
        --checkpoint_path /path/to/checkpoint \
        --port 5100
"""

import argparse
import asyncio
import base64
import io
import logging
import socket
import time
import traceback

import msgpack
import numpy as np
import torch
import websockets.asyncio.server
import websockets.frames
from PIL import Image

# ── msgpack numpy support (inline, no external dependency) ──────────────────

import functools


def _pack_array(obj):
    if isinstance(obj, np.ndarray) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_Packer = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


# ── Foreact Server ──────────────────────────────────────────────────────────


class ForeactServer:
    """WebSocket server wrapping VisualForesightPipeline for subgoal image prediction."""

    def __init__(
        self,
        pipeline,
        host: str = "0.0.0.0",
        port: int = 5100,
        guidance_scale: float = 4.5,
        image_guidance_scale: float = 1.5,
        num_inference_steps: int = 8,
        idle_timeout: int = -1,
    ):
        self._pipeline = pipeline
        self._host = host
        self._port = port
        self._guidance_scale = guidance_scale
        self._image_guidance_scale = image_guidance_scale
        self._num_inference_steps = num_inference_steps
        self._idle_timeout = idle_timeout
        self._last_active = time.time()
        self._request_count = 0
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self):
        asyncio.run(self._run())

    async def _run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            logging.info(f"Foreact server listening on {self._host}:{self._port}")
            if self._idle_timeout > 0:
                await self._idle_watchdog(server)
            else:
                await server.serve_forever()

    async def _idle_watchdog(self, server):
        while True:
            await asyncio.sleep(5)
            if time.time() - self._last_active > self._idle_timeout:
                logging.info(f"Idle timeout ({self._idle_timeout}s) reached, shutting down.")
                server.close()
                await server.wait_closed()
                break

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = _Packer()

        # Send metadata on connect (same protocol as starVLA server)
        metadata = {"service": "foreact", "status": "ready"}
        await websocket.send(packer.pack(metadata))

        while True:
            try:
                msg = _unpackb(await websocket.recv())
                self._last_active = time.time()
                ret = self._handle_request(msg)
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                tb = traceback.format_exc()
                logging.error(f"Error handling request:\n{tb}")
                await websocket.send(tb)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error.",
                )
                raise

    def _handle_request(self, msg: dict) -> dict:
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "predict")

        if mtype == "ping":
            return {"status": "ok", "type": "ping", "request_id": req_id}

        if mtype == "predict":
            return self._predict(msg, req_id)

        return {
            "status": "error",
            "request_id": req_id,
            "error": f"Unknown message type: {mtype}",
        }

    @torch.no_grad()
    def _predict(self, msg: dict, req_id: str) -> dict:
        """
        Predict subgoal image from current observation + task description.

        Expected msg keys:
            - image: np.ndarray (H, W, 3) uint8 RGB
            - task_description: str
            - guidance_scale: float (optional)
            - image_guidance_scale: float (optional)
            - num_inference_steps: int (optional)
            - seed: int (optional)
        """
        try:
            # Extract inputs
            image_arr = msg.get("image")
            if image_arr is None:
                return {
                    "status": "error",
                    "request_id": req_id,
                    "error": "Missing 'image' field",
                }

            task_description = msg.get("task_description", "")
            guidance_scale = msg.get("guidance_scale", self._guidance_scale)
            image_guidance_scale = msg.get("image_guidance_scale", self._image_guidance_scale)
            num_inference_steps = msg.get("num_inference_steps", self._num_inference_steps)
            seed = msg.get("seed", None)

            # Convert numpy array to PIL Image
            if isinstance(image_arr, np.ndarray):
                input_image = Image.fromarray(image_arr.astype(np.uint8))
            else:
                return {
                    "status": "error",
                    "request_id": req_id,
                    "error": f"Expected numpy array for 'image', got {type(image_arr)}",
                }

            # Set generator for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator(device="cuda").manual_seed(seed)

            self._request_count += 1
            t0 = time.time()
            logging.info(
                f"[Request #{self._request_count}] Predicting subgoal "
                f"(task='{task_description[:60]}', img={image_arr.shape}, "
                f"steps={num_inference_steps})"
            )

            # Run pipeline
            result = self._pipeline(
                image=input_image,
                caption=task_description,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=1,
                generator=generator,
            )

            # Convert output PIL image back to numpy array
            subgoal_image = result.images[0]
            subgoal_arr = np.array(subgoal_image, dtype=np.uint8)

            latency = time.time() - t0
            logging.info(
                f"[Request #{self._request_count}] Done in {latency:.2f}s, "
                f"output shape={subgoal_arr.shape}"
            )

            return {
                "status": "ok",
                "request_id": req_id,
                "data": {
                    "subgoal_image": subgoal_arr,
                    "latency": latency,
                },
            }

        except Exception as e:
            logging.exception(f"Prediction error (request_id={req_id})")
            return {
                "status": "error",
                "request_id": req_id,
                "error": str(e),
            }


def main():
    parser = argparse.ArgumentParser(description="Foreact Subgoal Image Prediction Server")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/media/raid/workspace/liyan/project/foreact/checkpoints/finetuned_bridge/run_finetune/checkpoint-7510",
        help="Path to model checkpoint",
    )
    parser.add_argument("--port", type=int, default=5100, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--num_inference_steps", type=int, default=8)
    parser.add_argument("--idle_timeout", type=int, default=1800)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s [%(levelname)s] %(message)s")

    # ── Load model ──
    import os
    import sys
    # Ensure foreact_2 modules are importable
    foreact_dir = os.path.dirname(os.path.abspath(__file__))
    if foreact_dir not in sys.path:
        sys.path.insert(0, foreact_dir)

    from pipeline import VisualForesightPipeline
    from utils.trainer_utils import find_newest_checkpoint

    ckpt = find_newest_checkpoint(args.checkpoint_path)
    logging.info(f"Loading model from {ckpt} ...")
    pipeline = VisualForesightPipeline.from_pretrained(
        ckpt,
        ignore_mismatched_sizes=True,
        _gradient_checkpointing=False,
        torch_dtype=torch.bfloat16,
    )
    pipeline = pipeline.to(device="cuda", dtype=torch.bfloat16)
    logging.info("Model loaded successfully!")

    # ── Start server ──
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(f"Host: {hostname}, IP: {local_ip}")

    server = ForeactServer(
        pipeline=pipeline,
        host=args.host,
        port=args.port,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        num_inference_steps=args.num_inference_steps,
        idle_timeout=args.idle_timeout,
    )
    server.serve_forever()


if __name__ == "__main__":
    import os
    main()
