import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_bridge_example() -> dict:
    """Creates a random input example for the Bridge policy."""
    return {
        "observation/image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/state": np.random.rand(8).astype(np.float32),
        "prompt": "pick up the object",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class BridgeInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)

        base_image = _parse_image(data["observation/image"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), np.zeros_like(base_image))
                image_masks = (np.True_, np.False_, np.False_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BridgeWithSubgoalInputs(BridgeInputs):
    """Bridge inputs with an extra future-GT subgoal image.

    Expects ``subgoal_images`` from the repack transform, with ``image_0``
    pointing to the aligned future-GT subgoal frame. The extra image is injected
    under ``subgoal_base_0_rgb`` so the VLM sees it as an additional visual
    context frame.
    """

    subgoal_source_key: str = "image_0"
    subgoal_model_key: str = "subgoal_base_0_rgb"

    def __call__(self, data: dict) -> dict:
        subgoal_images = data.pop("subgoal_images", None)
        result = super().__call__(data)

        if subgoal_images is None or self.subgoal_source_key not in subgoal_images:
            available = () if subgoal_images is None else tuple(subgoal_images)
            raise ValueError(
                f"Missing subgoal image '{self.subgoal_source_key}' in subgoal_images; available keys: {available}"
            )

        result["image"][self.subgoal_model_key] = _parse_image(subgoal_images[self.subgoal_source_key])
        result["image_mask"][self.subgoal_model_key] = np.True_
        return result
