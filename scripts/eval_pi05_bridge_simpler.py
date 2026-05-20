#!/usr/bin/env python3
"""Evaluate an OpenPI pi0.5 Bridge checkpoint in SimplerEnv.

This script runs inside the ``simpler_env`` conda environment and talks to an
OpenPI policy server over websocket.  It intentionally lives in this repository
so the SimplerEnv checkout does not need local source edits.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Mapping, Sequence
import dataclasses
import os
from pathlib import Path
import sys

import numpy as np


OPENPI_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIMPLER_ENV_ROOT = Path("/media/raid/workspace/xiahongyu/SimplerEnv")
DEFAULT_CKPT = OPENPI_ROOT / "checkpoints/pi05_bridge/bridge_reproduce/30000"
DEFAULT_LOGGING_DIR = OPENPI_ROOT / "results/simpler_eval/pi05_bridge_reproduce_30000"
DEFAULT_DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_DOUBAO_MODEL = "doubao-seed-2-0-pro-260215"
IMAGE_PREPROCESS_MODES = (
    "none",
    "center_crop_square",
    "resize_square_224",
    "resize_square_256",
    "center_crop_resize_224",
    "center_crop_resize_256",
)


def _insert_paths(simpler_env_root: Path) -> None:
    paths = [
        OPENPI_ROOT / "packages/openpi-client/src",
        OPENPI_ROOT,
        simpler_env_root,
        simpler_env_root / "ManiSkill2_real2sim",
    ]
    for path in reversed(paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


@dataclasses.dataclass(frozen=True)
class BridgeTaskSpec:
    task_key: str
    env_name: str
    scene_name: str
    robot: str
    rgb_overlay_path: str
    robot_init_x: float
    robot_init_y: float
    max_episode_steps: int


def bridge_task_specs(simpler_env_root: Path) -> list[BridgeTaskSpec]:
    real_inpainting = simpler_env_root / "ManiSkill2_real2sim/data/real_inpainting"
    return [
        BridgeTaskSpec(
            task_key="stack_green_cube_on_yellow_cube",
            env_name="StackGreenCubeOnYellowCubeBakedTexInScene-v0",
            scene_name="bridge_table_1_v1",
            robot="widowx",
            rgb_overlay_path=str(real_inpainting / "bridge_real_eval_1.png"),
            robot_init_x=0.147,
            robot_init_y=0.028,
            max_episode_steps=60,
        ),
        BridgeTaskSpec(
            task_key="put_carrot_on_plate",
            env_name="PutCarrotOnPlateInScene-v0",
            scene_name="bridge_table_1_v1",
            robot="widowx",
            rgb_overlay_path=str(real_inpainting / "bridge_real_eval_1.png"),
            robot_init_x=0.147,
            robot_init_y=0.028,
            max_episode_steps=60,
        ),
        BridgeTaskSpec(
            task_key="put_spoon_on_tablecloth",
            env_name="PutSpoonOnTableClothInScene-v0",
            scene_name="bridge_table_1_v1",
            robot="widowx",
            rgb_overlay_path=str(real_inpainting / "bridge_real_eval_1.png"),
            robot_init_x=0.147,
            robot_init_y=0.028,
            max_episode_steps=60,
        ),
        BridgeTaskSpec(
            task_key="put_eggplant_in_basket",
            env_name="PutEggplantInBasketScene-v0",
            scene_name="bridge_table_1_v2",
            robot="widowx_sink_camera_setup",
            rgb_overlay_path=str(real_inpainting / "bridge_sink.png"),
            robot_init_x=0.127,
            robot_init_y=0.06,
            max_episode_steps=120,
        ),
    ]


class Pi05BridgeServerPolicy:
    """Adapter from SimplerEnv images/states to OpenPI websocket actions."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        action_scale: float = 1.0,
        replan_steps: int = 5,
    ) -> None:
        from openpi_client import websocket_client_policy

        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self.action_scale = action_scale
        self.replan_steps = replan_steps
        self.task_description: str | None = None
        self.action_plan: deque[np.ndarray] = deque()
        self.current_open_gripper = 1.0
        self._last_plan_prompt: str | None = None

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.action_plan.clear()
        self.current_open_gripper = 1.0
        self._last_plan_prompt = None

    @staticmethod
    def _prompt_with_cot(task_description: str, cot_metadata: Mapping[str, object] | None) -> str:
        prompt = task_description
        if not cot_metadata:
            return prompt

        current_subtask = cot_metadata.get("current_subtask")
        if current_subtask is None:
            return prompt
        current_subtask = str(current_subtask).strip()
        if not current_subtask:
            return prompt

        return f"{prompt}, subtask: {current_subtask}"

    def step(
        self,
        image: np.ndarray,
        state: np.ndarray,
        task_description: str | None = None,
        cot_metadata: Mapping[str, object] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        from transforms3d.euler import euler2axangle

        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        if image.dtype != np.uint8:
            raise TypeError(f"Expected uint8 image, got {image.dtype}")
        if state.shape != (8,):
            raise ValueError(f"Expected Bridge state shape (8,), got {state.shape}")

        effective_prompt = self._prompt_with_cot(self.task_description or "", cot_metadata)
        if effective_prompt != self._last_plan_prompt:
            # Subtask changes should take effect immediately instead of waiting
            # for leftover actions sampled under the previous prompt.
            self.action_plan.clear()

        if not self.action_plan:
            obs = {
                "observation/image": image,
                "observation/state": state.astype(np.float32),
                "prompt": effective_prompt,
            }
            response = self.client.infer(obs)
            actions = np.asarray(response["actions"], dtype=np.float64)
            if actions.ndim == 1:
                actions = actions[None, :]
            if actions.shape[-1] < 7:
                raise ValueError(f"Policy returned action shape {actions.shape}; need at least 7 dims")

            n = min(self.replan_steps, actions.shape[0])
            self.action_plan.extend(actions[:n, :7])
            self._last_plan_prompt = effective_prompt

        raw = np.asarray(self.action_plan.popleft(), dtype=np.float64)
        open_gripper = np.asarray(raw[6:7], dtype=np.float64)

        raw_action = {
            "world_vector": np.asarray(raw[:3], dtype=np.float64),
            "rotation_delta": np.asarray(raw[3:6], dtype=np.float64),
            "open_gripper": open_gripper,
        }

        roll, pitch, yaw = raw_action["rotation_delta"]
        axis, angle = euler2axangle(roll, pitch, yaw)
        action = {
            "world_vector": raw_action["world_vector"] * self.action_scale,
            "rot_axangle": np.asarray(axis, dtype=np.float64) * angle * self.action_scale,
            # WidowX Bridge gripper controller uses +1=open, -1=close.
            "gripper": 2.0 * (open_gripper > 0.5).astype(np.float64) - 1.0,
            "terminate_episode": np.array([0.0], dtype=np.float64),
        }
        self.current_open_gripper = float(open_gripper[0] > 0.5)
        return raw_action, action

    def visualize_epoch(
        self,
        predicted_raw_actions: Sequence[dict[str, np.ndarray]],
        images: Sequence[np.ndarray],
        save_path: str,
    ) -> None:
        import cv2
        import matplotlib.pyplot as plt

        if not predicted_raw_actions:
            return

        resized_images = [cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA) for image in images]
        stride = max(1, len(resized_images) // 24)
        img_strip = np.concatenate(resized_images[::stride], axis=1)

        labels = ["x", "y", "z", "roll", "pitch", "yaw", "open_gripper"]
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ],
            dtype=np.float64,
        )

        fig, axes = plt.subplot_mosaic([["image"] * len(labels), labels], figsize=(3 * len(labels), 6))
        axes["image"].imshow(img_strip)
        axes["image"].set_axis_off()
        for i, label in enumerate(labels):
            axes[label].plot(pred_actions[:, i])
            axes[label].set_title(label)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)


def bridge_open_gripper_from_env(env, fallback_open_gripper: float) -> float:
    """Return Bridge-style open-gripper state from the simulator when available."""
    get_closedness = getattr(env.agent, "get_gripper_closedness", None)
    if get_closedness is None:
        return float(fallback_open_gripper)
    closedness = float(get_closedness())
    return float(np.clip(1.0 - closedness, 0.0, 1.0))


def bridge_state_from_env(env, fallback_open_gripper: float) -> np.ndarray:
    """Convert the current SimplerEnv WidowX TCP pose to Bridge dataset state."""
    from transforms3d.euler import mat2euler
    from transforms3d.quaternions import quat2mat

    # SimplerEnv's Bridge debug utilities use:
    #   tcp_rot = euler2mat(bridge_rpy) @ mat_transform
    # so evaluation needs the inverse transform.
    mat_transform = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    tcp_pose_at_robot_base = env.agent.robot.pose.inv() * env.tcp.pose
    bridge_rot = quat2mat(tcp_pose_at_robot_base.q) @ mat_transform.T
    roll, pitch, yaw = mat2euler(bridge_rot)
    open_gripper = bridge_open_gripper_from_env(env, fallback_open_gripper)
    state = np.array(
        [
            *tcp_pose_at_robot_base.p,
            roll,
            pitch,
            yaw,
            0.0,
            open_gripper,
        ],
        dtype=np.float32,
    )
    return state


def preprocess_policy_image(image: np.ndarray, mode: str) -> np.ndarray:
    """Adjust SimplerEnv RGB observations before sending them to the policy."""
    if mode == "none":
        return image
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")

    def center_crop_square(arr: np.ndarray) -> np.ndarray:
        height, width = arr.shape[:2]
        side = min(height, width)
        top = (height - side) // 2
        left = (width - side) // 2
        return arr[top : top + side, left : left + side]

    def resize_square(arr: np.ndarray, size: int) -> np.ndarray:
        import cv2

        interpolation = cv2.INTER_AREA if max(arr.shape[:2]) > size else cv2.INTER_LINEAR
        resized = cv2.resize(arr, (size, size), interpolation=interpolation)
        if resized.dtype != arr.dtype:
            resized = resized.astype(arr.dtype)
        return resized

    if mode == "center_crop_square":
        return center_crop_square(image).copy()
    if mode.startswith("resize_square_"):
        size = int(mode.removeprefix("resize_square_"))
        return resize_square(image, size)
    if mode.startswith("center_crop_resize_"):
        size = int(mode.removeprefix("center_crop_resize_"))
        return resize_square(center_crop_square(image), size)
    raise ValueError(f"Unknown image preprocess mode {mode!r}; valid modes: {IMAGE_PREPROCESS_MODES}")


def get_policy_image(env, obs, *, camera_name: str | None, image_preprocess: str) -> np.ndarray:
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=camera_name)
    return preprocess_policy_image(image, image_preprocess)


def run_single_episode(
    *,
    model: Pi05BridgeServerPolicy,
    ckpt_path: Path,
    spec: BridgeTaskSpec,
    obj_episode_id: int,
    logging_dir: Path,
    control_freq: int,
    sim_freq: int,
    obs_camera_name: str | None,
    additional_env_save_tags: str | None,
    enable_raytracing: bool,
    renderer_device: str,
    image_preprocess: str,
) -> bool:
    from transforms3d.euler import quat2euler

    from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
    from simpler_env.utils.visualization import write_video

    control_mode = get_robot_control_mode(spec.robot, "pi05")
    additional_env_build_kwargs = {}
    if enable_raytracing:
        additional_env_build_kwargs["shader_dir"] = "rt"

    renderer_kwargs = {"offscreen_only": True}
    if renderer_device:
        renderer_kwargs["device"] = renderer_device

    env = build_maniskill2_env(
        spec.env_name,
        obs_mode="rgbd",
        robot=spec.robot,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=spec.max_episode_steps,
        scene_name=spec.scene_name,
        renderer_kwargs=renderer_kwargs,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=spec.rgb_overlay_path,
        **additional_env_build_kwargs,
    )

    images: list[np.ndarray] = []
    predicted_actions: list[dict[str, np.ndarray]] = []
    info = {}
    success = "failure"
    robot_init_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    try:
        obs, _ = env.reset(
            options={
                "robot_init_options": {
                    "init_xy": np.array([spec.robot_init_x, spec.robot_init_y]),
                    "init_rot_quat": robot_init_quat,
                },
                "obj_init_options": {"episode_id": obj_episode_id},
            }
        )
        is_final_subtask = env.is_final_subtask()
        task_description = env.get_language_instruction()
        print(f"[Episode {obj_episode_id:03d}] {spec.task_key}: {task_description}", flush=True)

        model.reset(task_description)
        image = get_policy_image(env, obs, camera_name=obs_camera_name, image_preprocess=image_preprocess)
        images.append(image)

        predicted_terminated = False
        truncated = False
        timestep = 0
        while not (predicted_terminated or truncated):
            state = bridge_state_from_env(env, model.current_open_gripper)
            raw_action, action = model.step(image, state, task_description)
            predicted_actions.append(raw_action)
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            if predicted_terminated and not is_final_subtask:
                predicted_terminated = False
                env.advance_to_next_subtask()

            obs, _reward, done, truncated, info = env.step(
                np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
            )

            if done:
                success = "success"
                image = get_policy_image(env, obs, camera_name=obs_camera_name, image_preprocess=image_preprocess)
                images.append(image)
                timestep += 1
                for _ in range(10):
                    state = bridge_state_from_env(env, model.current_open_gripper)
                    raw_action, action = model.step(image, state, task_description)
                    predicted_actions.append(raw_action)
                    obs, _reward, _done_extra, truncated_extra, info = env.step(
                        np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
                    )
                    image = get_policy_image(env, obs, camera_name=obs_camera_name, image_preprocess=image_preprocess)
                    images.append(image)
                    timestep += 1
                    if truncated_extra:
                        break
                break

            new_task_description = env.get_language_instruction()
            if new_task_description != task_description:
                task_description = new_task_description
                model.reset(task_description)
                print(f"[Episode {obj_episode_id:03d}] New instruction: {task_description}", flush=True)
            is_final_subtask = env.is_final_subtask()

            if timestep % 10 == 0:
                print(f"[Episode {obj_episode_id:03d}] step={timestep} info={info}", flush=True)

            image = get_policy_image(env, obs, camera_name=obs_camera_name, image_preprocess=image_preprocess)
            images.append(image)
            timestep += 1
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()

    episode_stats = info.get("episode_stats", {})

    env_save_name = spec.env_name
    for key, value in additional_env_build_kwargs.items():
        env_save_name = f"{env_save_name}_{key}_{value}"
    if additional_env_save_tags:
        env_save_name = f"{env_save_name}_{additional_env_save_tags}"
    if image_preprocess != "none":
        env_save_name = f"{env_save_name}_image_{image_preprocess}"

    ckpt_basename = ckpt_path.rstrip("/").split("/")[-1] if isinstance(ckpt_path, str) else ckpt_path.name
    video_name = f"{success}_obj_episode_{obj_episode_id}"
    for key, value in episode_stats.items():
        video_name = f"{video_name}_{key}_{value}"
    video_name = f"{video_name}.mp4"

    overlay_name = Path(spec.rgb_overlay_path).stem if spec.rgb_overlay_path else "None"
    r, p, y = quat2euler(robot_init_quat)
    video_path = (
        logging_dir
        / ckpt_basename
        / spec.scene_name
        / control_mode
        / env_save_name
        / f"rob_{spec.robot_init_x}_{spec.robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{overlay_name}"
        / video_name
    )
    write_video(str(video_path), images, fps=5)

    action_path = video_path.with_suffix(".png")
    action_path = action_path.parent / "actions" / action_path.name
    model.visualize_epoch(predicted_actions, images, str(action_path))

    print(f"[Episode {obj_episode_id:03d}] {success}; video={video_path}", flush=True)
    return success == "success"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simpler-env-root", type=Path, default=DEFAULT_SIMPLER_ENV_ROOT)
    parser.add_argument("--ckpt-path", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--logging-dir", type=Path, default=DEFAULT_LOGGING_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--task", default="all", help="Task key to run, or 'all'.")
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--control-freq", type=int, default=5)
    parser.add_argument("--sim-freq", type=int, default=500)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--obs-camera-name", default=None)
    parser.add_argument("--additional-env-save-tags", default="pi05_bridge_openpi")
    parser.add_argument(
        "--image-preprocess",
        choices=IMAGE_PREPROCESS_MODES,
        default="resize_square_224",
        help=(
            "Preprocess RGB observations before policy inference. "
            "Use center_crop_square to remove 480x640 aspect-ratio padding before OpenPI resize_with_pad."
        ),
    )
    parser.add_argument(
        "--enable-doubao-subtask",
        action="store_true",
        help="Use eval_cot_manager.py to generate online subtasks and append them to the policy prompt.",
    )
    parser.add_argument(
        "--doubao-api-key",
        default=os.environ.get("DOUBAO_API_KEY") or os.environ.get("ARK_API_KEY") or os.environ.get("VOLCENKEY"),
        help="Doubao/Ark API key. Defaults to DOUBAO_API_KEY, ARK_API_KEY, then VOLCENKEY.",
    )
    parser.add_argument("--doubao-base-url", default=DEFAULT_DOUBAO_BASE_URL)
    parser.add_argument("--doubao-model", default=DEFAULT_DOUBAO_MODEL)
    parser.add_argument("--cot-refresh-interval", type=int, default=6)
    parser.add_argument(
        "--cot-log-dir",
        type=Path,
        default=None,
        help="Directory for Doubao subtask image/text logs. Defaults to <logging-dir>/cot_logs.",
    )
    parser.add_argument("--enable-raytracing", action="store_true")
    parser.add_argument(
        "--renderer-device",
        default="",
        help="Optional SAPIEN offscreen renderer device. Empty lets SAPIEN choose the Vulkan device.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simpler_env_root = args.simpler_env_root.resolve()
    _insert_paths(simpler_env_root)

    # SimplerEnv asset paths are relative in several places, so run from its root.
    os.chdir(simpler_env_root)
    os.environ["DISPLAY"] = ""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    specs = bridge_task_specs(simpler_env_root)
    if args.task != "all":
        specs = [spec for spec in specs if spec.task_key == args.task or spec.env_name == args.task]
        if not specs:
            valid = ", ".join(spec.task_key for spec in bridge_task_specs(simpler_env_root))
            raise ValueError(f"Unknown task {args.task!r}; valid tasks: all, {valid}")

    args.logging_dir.mkdir(parents=True, exist_ok=True)
    print(f"Python: {sys.executable}", flush=True)
    print(f"SimplerEnv root: {simpler_env_root}", flush=True)
    print(f"Checkpoint: {args.ckpt_path}", flush=True)
    print(f"Logging dir: {args.logging_dir}", flush=True)
    print(f"Tasks: {[spec.task_key for spec in specs]}", flush=True)
    print(f"Episodes per task: {args.episodes}", flush=True)
    print(f"Image preprocess: {args.image_preprocess}", flush=True)
    print(f"Doubao subtask: {args.enable_doubao_subtask}", flush=True)

    model = Pi05BridgeServerPolicy(
        host=args.host,
        port=args.port,
        action_scale=args.action_scale,
        replan_steps=args.replan_steps,
    )
    if args.enable_doubao_subtask:
        if not args.doubao_api_key:
            raise ValueError("Doubao subtask eval requires --doubao-api-key or DOUBAO_API_KEY/ARK_API_KEY/VOLCENKEY.")
        from eval_cot_manager import DoubaoVLM, EvalCoTManager

        cot_log_dir = args.cot_log_dir or (args.logging_dir / "cot_logs")
        vlm = DoubaoVLM(
            api_key=args.doubao_api_key,
            base_url=args.doubao_base_url,
            model=args.doubao_model,
            log_dir=str(cot_log_dir),
        )
        model = EvalCoTManager(model, vlm, cot_refresh_interval=args.cot_refresh_interval)
        print(f"CoT refresh interval: {args.cot_refresh_interval}", flush=True)
        print(f"CoT log dir: {cot_log_dir}", flush=True)

    all_results: dict[str, list[bool]] = {}
    for spec in specs:
        results: list[bool] = []
        end_episode = args.start_episode + args.episodes
        print(f"[Task {spec.task_key}] Running object episodes {args.start_episode}:{end_episode}", flush=True)
        for obj_episode_id in range(args.start_episode, end_episode):
            result = run_single_episode(
                model=model,
                ckpt_path=args.ckpt_path,
                spec=spec,
                obj_episode_id=obj_episode_id,
                logging_dir=args.logging_dir,
                control_freq=args.control_freq,
                sim_freq=args.sim_freq,
                obs_camera_name=args.obs_camera_name,
                additional_env_save_tags=args.additional_env_save_tags,
                enable_raytracing=args.enable_raytracing,
                renderer_device=args.renderer_device,
                image_preprocess=args.image_preprocess,
            )
            results.append(result)
            print(
                f"[Task {spec.task_key}] Progress {len(results)}/{args.episodes}; "
                f"success_rate={np.mean(results):.4f}",
                flush=True,
            )
        all_results[spec.task_key] = results
        print(f"[Task {spec.task_key}] Average success {np.mean(results):.4f}", flush=True)

    total = [value for results in all_results.values() for value in results]
    print("=== Summary ===", flush=True)
    for task_key, results in all_results.items():
        print(f"{task_key}: {sum(results)}/{len(results)} = {np.mean(results):.4f}", flush=True)
    print(f"overall: {sum(total)}/{len(total)} = {np.mean(total):.4f}", flush=True)


if __name__ == "__main__":
    main()
