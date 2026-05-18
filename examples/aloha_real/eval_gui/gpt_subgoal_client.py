"""GPT-based subgoal image generation client.

Uses the image generation API at https://www.right.codes/draw to produce
subgoal images from a reference camera frame and task description.

The client exposes the same ``predict_subgoal(image, task)`` interface as
``ForeactClient`` so it can be used as a drop-in replacement in SubgoalMode.
"""
from __future__ import annotations

import base64
import io
import logging
import threading
import time
from typing import Optional

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://www.right.codes/draw"
_DEFAULT_MODEL = "gpt-image-2"
_DEFAULT_SIZE = "1024x1024"
_REQUEST_TIMEOUT = 120.0


class GptSubgoalClient:
    """Thread-safe client that generates subgoal images via GPT image API."""
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        model: str = _DEFAULT_MODEL,
        size: str = _DEFAULT_SIZE,
        timeout: float = _REQUEST_TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._size = size
        self._timeout = timeout
        self._lock = threading.Lock()
        self._session = requests.Session()
        self._custom_prompt: Optional[str] = None

    def connect(self) -> dict:
        return {"client": "gpt_subgoal", "model": self._model}

    def close(self) -> None:
        self._session.close()


    def build_prompt(prompt):
        """构建图像生成的完整 prompt"""
        return (
            f"This image depicts a robotic arm manipulation scene. The overall task is: \"{prompt}\".\n"
            f"First, explicitly analyze the spatial geometry. Then, deduce ONLY the **[Next Micro-Action]**.\n"
            f"\n"
            f"### Step 1: Spatial Perception (MANDATORY)\n"
            f"Before taking any action, you MUST output the following geometric analysis:\n"
            f"1. Object Axis: Describe the orientation of the target object's longest axis.\n"
            f"2. Gripper Axis: Describe the current orientation of the gripper's opening.\n"
            f"3. Alignment Status: Are the Gripper Axis and Object Axis strictly parallel? (Answer: Yes / No)\n"
            f"\n"
            f"### Step 2: Action Deduction Logic (State-Based)\n"
            f"Apply the corresponding single micro-action based on Step 1:\n"
            f"- If approaching: Move the gripper directly above or beside the object.\n"
            f"- If near but Alignment Status is NO: You are STRICTLY FORBIDDEN to move closer or close the fingers. "
            f"The ONLY permitted action is to **ROTATE the gripper around its Z-axis (yaw)**. "
            f"Specify the exact rotational adjustment (e.g., 'Rotate clockwise by approximately 30 degrees') "
            f"to align with the Object Axis.\n"
            f"- If near and Alignment Status is YES: Close the metallic fingers inward from both sides "
            f"until their inner surfaces *just* touch the object.\n"
            f"- If securely grasped: Move the gripper and object together toward the destination.\n"
            f"\n"
            f"### Strict Physical Constraints (CRITICAL)\n"
            f"1. Rigid Body & Zero Clipping: Both the gripper and objects are non-deformable. "
            f"Clipping or intersecting is STRICTLY PROHIBITED.\n"
            f"2. Morphological Consistency: Maintain the gripper's exact mechanical structure. "
            f"No mutation or extra parts.\n"
            f"3. Environmental Stability: Background and ungrasped objects remain totally frozen. "
            f"No floating objects.\n"
        )

    def predict_subgoal(
        self,
        image: np.ndarray,
        task_description: str,
    ) -> Optional[np.ndarray]:
        """Generate a subgoal image using GPT image generation API.

        Takes an HxWx3 uint8 RGB reference image and task description,
        returns an HxWx3 uint8 RGB subgoal image resized to match the input.
        """
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        assert image.ndim == 3 and image.shape[2] == 3

        ref_b64 = self._encode_image_base64(image)
        if self._custom_prompt:
            prompt = self._custom_prompt.replace("{task}", task_description)
        else:
            prompt = self.build_prompt(task_description)

        url = f"{self._base_url}/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "prompt": prompt,
            "image": [f"data:image/jpeg;base64,{ref_b64}"],
            "size": self._size,
            "response_format": "url",
        }

        t0 = time.time()
        with self._lock:
            try:
                resp = self._session.post(
                    url, json=payload, headers=headers, timeout=self._timeout
                )
                resp.raise_for_status()
            except requests.RequestException as e:
                logger.error("GPT subgoal API request failed: %s", e)
                return None

        data = resp.json()
        image_url = self._extract_image_url(data)
        if image_url is None:
            logger.error("GPT subgoal API returned no image URL: %s", data)
            return None

        subgoal = self._download_image(image_url, target_shape=image.shape[:2])
        if subgoal is not None:
            logger.debug(
                "GPT subgoal generated: shape=%s, total=%.2fs",
                subgoal.shape, time.time() - t0,
            )
        return subgoal

    @staticmethod
    def _encode_image_base64(image: np.ndarray) -> str:
        pil_img = Image.fromarray(image)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _extract_image_url(response_data: dict) -> Optional[str]:
        try:
            return response_data["data"][0]["url"]
        except (KeyError, IndexError, TypeError):
            return None

    def _download_image(
        self, url: str, target_shape: tuple[int, int]
    ) -> Optional[np.ndarray]:
        try:
            resp = self._session.get(url, timeout=self._timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error("Failed to download GPT subgoal image: %s", e)
            return None

        try:
            pil_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            target_h, target_w = target_shape
            pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)
            return np.asarray(pil_img, dtype=np.uint8)
        except Exception as e:
            logger.error("Failed to decode GPT subgoal image: %s", e)
            return None
