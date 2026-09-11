"""Native Piper observations: no Trossen joint flips or gripper conversions."""

import dataclasses

import numpy as np

from openpi import transforms

CAMERA_MAP = {
    "cam_high": "base_0_rgb",
    "cam_left_wrist": "left_wrist_0_rgb",
    "cam_right_wrist": "right_wrist_0_rgb",
}
JOINT_MASK = transforms.make_bool_mask(6, -1, 6, -1)


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """Accept three RGB cameras, native absolute state and optional actions.

    Float images use CHW/HWC [0, 1]; uint8 images use CHW/HWC [0, 255].
    Missing cameras are errors for this dataset rather than silent black inputs.
    Supervision and episode metadata deliberately do not enter the observation.
    """

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError(f"Expected finite Piper state [14], got {state.shape}")
        images = {}
        for camera, destination in CAMERA_MAP.items():
            image = np.asarray(data["images"][camera])
            if image.ndim != 3:
                raise ValueError(f"{camera}: expected rank-3 RGB image")
            if image.shape[0] == 3:
                image = np.moveaxis(image, 0, -1)
            if image.shape[-1] != 3:
                raise ValueError(f"{camera}: expected RGB, got {image.shape}")
            if np.issubdtype(image.dtype, np.floating):
                if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                    raise ValueError(f"{camera}: float RGB must be finite and in [0, 1]")
                image = np.rint(image * 255).astype(np.uint8)
            elif image.dtype != np.uint8:
                raise ValueError(f"{camera}: expected uint8 or floating RGB")
            images[destination] = image
        result = {"state": state.copy(), "image": images, "image_mask": dict.fromkeys(images, np.True_)}
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 14 or not np.isfinite(actions).all():
                raise ValueError(f"Expected finite Piper actions [H, 14], got {actions.shape}")
            result["actions"] = actions.copy()
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < 14 or not np.isfinite(actions).all():
            raise ValueError("Expected finite model actions [H, D>=14]")
        return {"actions": actions[:, :14].copy()}
