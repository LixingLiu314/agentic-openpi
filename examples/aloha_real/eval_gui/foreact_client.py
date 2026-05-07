"""WebSocket client for the ForeAct subgoal-image prediction server.

Wire format matches ``server_foreact.py``: msgpack-encoded dicts with numpy
ndarray support. On connect the server pushes a metadata frame; subsequent
exchanges are request->response pairs.

Request schema (``predict``)::

    {
      "type": "predict",
      "request_id": <str>,
      "image": np.ndarray (H, W, 3) uint8 RGB,
      "task_description": <str>,
      # optional pipeline overrides
      "guidance_scale": float,
      "image_guidance_scale": float,
      "num_inference_steps": int,
      "seed": int | None,
    }

Response::

    {
      "status": "ok" | "error",
      "request_id": <str>,
      "data": {"subgoal_image": np.ndarray (H, W, 3) uint8, "latency": float},
      "error": <str>  # only on status=error
    }
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Optional

import numpy as np
import websockets.sync.client

from openpi_client import msgpack_numpy

logger = logging.getLogger(__name__)


class ForeactClient:
    """Thread-safe synchronous client for the ForeAct subgoal server."""

    def __init__(
        self,
        host: str = "10.1.119.68",
        port: int = 5100,
        connect_timeout: float = 30.0,
        request_timeout: float = 60.0,
    ) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._request_timeout = request_timeout
        self._connect_timeout = connect_timeout
        self._lock = threading.Lock()
        self._ws = None
        self._metadata: Optional[dict] = None

    def connect(self) -> dict:
        deadline = time.time() + self._connect_timeout
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            try:
                self._ws = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None
                )
                self._metadata = msgpack_numpy.unpackb(self._ws.recv())
                logger.info("ForeactClient connected: %s, metadata=%s", self._uri, self._metadata)
                return self._metadata
            except (ConnectionRefusedError, OSError) as e:
                last_err = e
                time.sleep(2.0)
        raise RuntimeError(f"Could not connect to ForeAct server at {self._uri}: {last_err}")

    def close(self) -> None:
        with self._lock:
            if self._ws is not None:
                try:
                    self._ws.close()
                except Exception:
                    pass
                self._ws = None

    def predict_subgoal(
        self,
        image: np.ndarray,
        task_description: str,
        *,
        guidance_scale: Optional[float] = None,
        image_guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Send an HxWx3 uint8 RGB image, return the predicted HxWx3 uint8 subgoal."""
        if self._ws is None:
            self.connect()

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        assert image.ndim == 3 and image.shape[2] == 3, f"Expected HWC RGB, got {image.shape}"

        msg: dict = {
            "type": "predict",
            "request_id": uuid.uuid4().hex,
            "image": image,
            "task_description": task_description,
        }
        if guidance_scale is not None:
            msg["guidance_scale"] = float(guidance_scale)
        if image_guidance_scale is not None:
            msg["image_guidance_scale"] = float(image_guidance_scale)
        if num_inference_steps is not None:
            msg["num_inference_steps"] = int(num_inference_steps)
        if seed is not None:
            msg["seed"] = int(seed)

        with self._lock:
            t0 = time.time()
            self._ws.send(self._packer.pack(msg))
            response = self._ws.recv(timeout=self._request_timeout)
            if isinstance(response, str):
                logger.error("ForeAct server returned error string: %s", response[:500])
                return None
            reply = msgpack_numpy.unpackb(response)

        if reply.get("status") != "ok":
            logger.error("ForeAct error: %s", reply.get("error"))
            return None

        sg = reply["data"]["subgoal_image"]
        logger.debug(
            "ForeAct predict_subgoal: shape=%s, server_latency=%.2fs, total=%.2fs",
            sg.shape, reply["data"].get("latency", -1), time.time() - t0,
        )
        return np.asarray(sg, dtype=np.uint8)
