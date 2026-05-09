#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    from colorama import Fore
    from colorama import Style
    from colorama import init as colorama_init

    colorama_init(autoreset=True)
except Exception:  # pragma: no cover - color fallback is intentionally simple.
    class _Fore:
        BLACK = "\033[30m"
        RED = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        MAGENTA = "\033[35m"
        CYAN = "\033[36m"
        WHITE = "\033[37m"
        RESET = "\033[39m"
        LIGHTGREEN_EX = "\033[92m"
        LIGHTCYAN_EX = "\033[96m"
        LIGHTYELLOW_EX = "\033[93m"

    class _Style:
        BRIGHT = "\033[1m"
        DIM = "\033[2m"
        NORMAL = "\033[22m"
        RESET_ALL = "\033[0m"

    Fore = _Fore()
    Style = _Style()


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
CALIB_DIR = TOOLS_DIR / "calibration"
CALIB_DATA_DIR = CALIB_DIR / "data"
TRAJ_DIR = TOOLS_DIR / "trajectory"
FK_MODULE = TRAJ_DIR / "mobile_aloha_link6_fk.py"

FRONT_CAMERA_TOPIC = "/camera_f/color/image_raw"
FRONT_INTRINSIC_IMAGES_DIR = CALIB_DATA_DIR / "front_intrinsic_imgs_charuco"
FRONT_INTRINSIC_PREFIX = "front_charuco"
FRONT_INTRINSIC_OUTPUT = CALIB_DATA_DIR / "intrinsics_front_charuco.json"
CHARUCO_ARGS = (
    "--target-type",
    "charuco",
    "--charuco-squares-x",
    "4",
    "--charuco-squares-y",
    "5",
    "--square-size",
    "0.05",
    "--marker-length",
    "0.037",
    "--aruco-dict",
    "DICT_4X4_50",
)


@dataclass
class WizardSession:
    front_intrinsics_ready: bool = False


@dataclass(frozen=True)
class ArmConfig:
    arm: str
    label_cn: str
    wrist_topic: str
    joint_topic: str
    intrinsic_images_dir: Path
    intrinsic_prefix: str
    intrinsic_output: Path
    handeye_images_dir: Path
    handeye_joints_path: Path
    handeye_manifest_path: Path
    handeye_prefix: str
    handeye_output: Path
    transfer_front_dir: Path
    transfer_wrist_dir: Path
    transfer_joints_path: Path
    transfer_manifest_path: Path
    transfer_prefix: str
    front_extrinsic_output: Path
    final_targets: tuple[Path, ...]


def paint(text: str, color: str) -> str:
    return f"{color}{text}{Style.RESET_ALL}"


def print_info(text: str) -> None:
    print(paint(text, Fore.CYAN), flush=True)


def print_success(text: str) -> None:
    print(paint(text, Fore.LIGHTGREEN_EX), flush=True)


def print_warning(text: str) -> None:
    print(paint(text, Fore.LIGHTYELLOW_EX), flush=True)


def print_error(text: str) -> None:
    print(paint(text, Fore.RED), flush=True)


def repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def is_non_empty_directory(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def prompt_line(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit(130) from None


def wait_for_enter(prompt: str) -> None:
    while True:
        reply = prompt_line(prompt)
        if reply == "":
            return
        print_warning("只需直接按 Enter 键继续，或者按 Ctrl+C 退出。")


def prompt_yes_no(prompt: str) -> bool:
    while True:
        reply = prompt_line(f"{prompt} ").lower()
        if reply in {"y", "yes"}:
            return True
        if reply in {"n", "no"}:
            return False
        print_warning("请输入 y 或 n。")


def prompt_menu_choice() -> str:
    while True:
        reply = prompt_line(
            "\n"
            + paint("请选择一个操作：", Fore.LIGHTCYAN_EX)
            + "\n"
            + "  1. 执行完整的左臂校准全流程\n"
            + "  2. 执行完整的右臂校准全流程\n"
            + "  3. 退出\n"
            + "请输入选项 1/2/3: "
        )
        if reply in {"1", "2", "3"}:
            return reply
        print_warning("无效输入，请输入 1、2 或 3。")


def run_command(
    command: Sequence[str],
    *,
    description: str,
    capture_output: bool,
    announce_command: bool = True,
) -> subprocess.CompletedProcess[str]:
    display = shlex.join(command)
    if announce_command:
        print_info(f"执行：{display}")
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=capture_output,
            check=False,
        )
    except KeyboardInterrupt:
        print_warning("\n已收到中断信号，当前步骤已停止。")
        raise

    if result.returncode != 0:
        print_error(f"{description} 失败，退出码：{result.returncode}")
        if capture_output:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        raise RuntimeError(description)
    return result


def maybe_warn_non_empty(path: Path, label: str) -> None:
    if is_non_empty_directory(path):
        print_warning(f"{label} 目录已经存在且非空：{path}")
        print_warning("继续执行会把该目录中的旧图片一并纳入后续解算。")
        if not prompt_yes_no("是否仍然继续本步骤？(y/n)"):
            raise SystemExit(1)


def capture_step(
    *,
    guide_lines: Sequence[str],
    command: Sequence[str],
    capture_dir: Path | None = None,
    capture_label: str | None = None,
) -> None:
    print()
    for line in guide_lines:
        print_info(line)
    if capture_dir is not None and capture_label is not None:
        maybe_warn_non_empty(capture_dir, capture_label)
    wait_for_enter("按 Enter 键启动相机采集...")
    run_command(command, description="采集步骤", capture_output=False)
    if not prompt_yes_no("采集是否顺利？是否继续进行下一步解算？(y/n)"):
        raise SystemExit(1)


def solve_step(
    *,
    guide_lines: Sequence[str],
    command: Sequence[str],
    output_path: Path,
    step_name: str,
) -> None:
    print()
    for line in guide_lines:
        print_info(line)
    wait_for_enter("按 Enter 键开始静默解算...")
    print_info("程序已启动，正在静默运行中，请稍候...")
    run_command(command, description=step_name, capture_output=True, announce_command=False)
    if not output_path.exists():
        raise RuntimeError(f"{step_name} 已完成，但未找到输出文件：{output_path}")
    print_success(f"{step_name} 成功：{output_path}")


def copy_final_outputs(source: Path, targets: Sequence[Path]) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(f"找不到源文件：{source}")

    written: list[Path] = []
    for target in targets:
        ensure_parent(target)
        if source.resolve() == target.resolve():
            written.append(target)
            continue
        shutil.copy2(source, target)
        written.append(target)
    return written


def build_arm_config(arm: str) -> ArmConfig:
    if arm == "left":
        return ArmConfig(
            arm="left",
            label_cn="左臂",
            wrist_topic="/camera_l/color/image_raw",
            joint_topic="/puppet/joint_left",
            intrinsic_images_dir=CALIB_DATA_DIR / "wrist_left_intrinsic_imgs_charuco",
            intrinsic_prefix="wrist_left_charuco",
            intrinsic_output=CALIB_DATA_DIR / "intrinsics_wrist_left_charuco.json",
            handeye_images_dir=CALIB_DATA_DIR / "wrist_left_eye_in_hand_imgs_charuco",
            handeye_joints_path=CALIB_DATA_DIR / "wrist_left_eye_in_hand_joints_charuco.npy",
            handeye_manifest_path=CALIB_DATA_DIR / "wrist_left_eye_in_hand_manifest_charuco.json",
            handeye_prefix="wrist_left_eih_charuco",
            handeye_output=CALIB_DATA_DIR / "wrist_left_eye_in_hand_charuco_link6.json",
            transfer_front_dir=CALIB_DATA_DIR / "transfer_front_left_charuco" / "front_imgs",
            transfer_wrist_dir=CALIB_DATA_DIR / "transfer_front_left_charuco" / "wrist_imgs",
            transfer_joints_path=CALIB_DATA_DIR / "transfer_front_left_charuco" / "joints.npy",
            transfer_manifest_path=CALIB_DATA_DIR / "transfer_front_left_charuco" / "manifest.json",
            transfer_prefix="transfer_left_charuco",
            front_extrinsic_output=CALIB_DATA_DIR / "front_via_left_wrist_charuco_link6.json",
            final_targets=(CALIB_DATA_DIR / "front_in_left_base_from_left_charuco_final.json",),
        )
    if arm == "right":
        return ArmConfig(
            arm="right",
            label_cn="右臂",
            wrist_topic="/camera_r/color/image_raw",
            joint_topic="/puppet/joint_right",
            intrinsic_images_dir=CALIB_DATA_DIR / "wrist_right_intrinsic_imgs_charuco",
            intrinsic_prefix="wrist_right_charuco",
            intrinsic_output=CALIB_DATA_DIR / "intrinsics_wrist_right_charuco.json",
            handeye_images_dir=CALIB_DATA_DIR / "wrist_right_eye_in_hand_imgs_charuco",
            handeye_joints_path=CALIB_DATA_DIR / "wrist_right_eye_in_hand_joints_charuco.npy",
            handeye_manifest_path=CALIB_DATA_DIR / "wrist_right_eye_in_hand_manifest_charuco.json",
            handeye_prefix="wrist_right_eih_charuco",
            handeye_output=CALIB_DATA_DIR / "wrist_right_eye_in_hand_charuco_link6.json",
            transfer_front_dir=CALIB_DATA_DIR / "transfer_front_right_charuco" / "front_imgs",
            transfer_wrist_dir=CALIB_DATA_DIR / "transfer_front_right_charuco" / "wrist_imgs",
            transfer_joints_path=CALIB_DATA_DIR / "transfer_front_right_charuco" / "joints.npy",
            transfer_manifest_path=CALIB_DATA_DIR / "transfer_front_right_charuco" / "manifest.json",
            transfer_prefix="transfer_right_charuco",
            front_extrinsic_output=CALIB_DATA_DIR / "front_via_right_wrist_charuco_link6.json",
            final_targets=(
                CALIB_DATA_DIR / "front_in_right_base_from_left_charuco_final.json",
                CALIB_DATA_DIR / "front_in_right_base_from_right_charuco_final.json",
            ),
        )
    raise ValueError(f"Unsupported arm: {arm}")


def build_intrinsic_capture_cmd(image_topic: str, output_dir: Path, prefix: str) -> list[str]:
    return [
        sys.executable,
        repo_rel(CALIB_DIR / "capture_front_calib_images.py"),
        "--topic",
        image_topic,
        "--output-dir",
        repo_rel(output_dir),
        "--prefix",
        prefix,
        "--show",
    ]


def build_intrinsic_solve_cmd(images_dir: Path, output: Path) -> list[str]:
    cmd = [
        sys.executable,
        repo_rel(CALIB_DIR / "calib_intrinsic.py"),
        "--images-dir",
        repo_rel(images_dir),
        *CHARUCO_ARGS,
        "--output",
        repo_rel(output),
    ]
    return cmd


def build_front_intrinsic_capture_cmd() -> list[str]:
    return build_intrinsic_capture_cmd(
        FRONT_CAMERA_TOPIC,
        FRONT_INTRINSIC_IMAGES_DIR,
        FRONT_INTRINSIC_PREFIX,
    )


def build_front_intrinsic_solve_cmd() -> list[str]:
    return build_intrinsic_solve_cmd(FRONT_INTRINSIC_IMAGES_DIR, FRONT_INTRINSIC_OUTPUT)


def ensure_front_intrinsics(session: WizardSession) -> Path:
    if session.front_intrinsics_ready and FRONT_INTRINSIC_OUTPUT.exists():
        return FRONT_INTRINSIC_OUTPUT

    print()
    print_success("主视角相机（前置相机）内参标定")
    print_info("步骤 0/6：先完成主视角相机内参采集与解算。")
    if FRONT_INTRINSIC_OUTPUT.exists():
        print_info(
            "检测到已有的前置相机内参文件："
            f"{repo_rel(FRONT_INTRINSIC_OUTPUT)}"
        )
        if not prompt_yes_no("是否重新标定主视角相机内参？(y/n)"):
            session.front_intrinsics_ready = True
            print_success("将直接使用已有的主视角相机内参。")
            return FRONT_INTRINSIC_OUTPUT
        print_warning("你选择了重新标定主视角相机内参。")
    else:
        print_warning(
            "未找到主视角相机内参文件，向导将先完成前置相机内参采集与解算。"
        )

    capture_step(
        guide_lines=(
            "步骤 0/6：采集主视角相机内参图片。",
            "请手持标定板在主视角相机前缓慢移动，尽量覆盖画面中心、四角和不同距离。",
            "请保持标定板清晰对焦，尽量避免运动模糊，并多拍几张不同角度的图像。",
        ),
        command=build_front_intrinsic_capture_cmd(),
        capture_dir=FRONT_INTRINSIC_IMAGES_DIR,
        capture_label="主视角相机内参采集目录",
    )

    solve_step(
        guide_lines=(
            "步骤 0/6：静默解算主视角相机内参。",
            "程序将根据刚才采集的主视角相机图片自动生成相机内参 JSON。",
        ),
        command=build_front_intrinsic_solve_cmd(),
        output_path=FRONT_INTRINSIC_OUTPUT,
        step_name="主视角相机内参解算",
    )

    session.front_intrinsics_ready = True
    return FRONT_INTRINSIC_OUTPUT


def build_handeye_capture_cmd(cfg: ArmConfig) -> list[str]:
    joint_flag = "--joint-topic-left" if cfg.arm == "left" else "--joint-topic-right"
    return [
        sys.executable,
        repo_rel(CALIB_DIR / "capture_front_handeye_samples.py"),
        "--arm",
        cfg.arm,
        "--topic",
        cfg.wrist_topic,
        joint_flag,
        cfg.joint_topic,
        "--output-dir",
        repo_rel(cfg.handeye_images_dir),
        "--joints-path",
        repo_rel(cfg.handeye_joints_path),
        "--manifest-path",
        repo_rel(cfg.handeye_manifest_path),
        "--prefix",
        cfg.handeye_prefix,
        "--show",
    ]


def build_handeye_solve_cmd(cfg: ArmConfig) -> list[str]:
    return [
        sys.executable,
        repo_rel(CALIB_DIR / "solve_wrist_eye_in_hand.py"),
        "--images-dir",
        repo_rel(cfg.handeye_images_dir),
        "--joints",
        repo_rel(cfg.handeye_joints_path),
        "--intrinsics",
        repo_rel(cfg.intrinsic_output),
        "--fk-module",
        repo_rel(FK_MODULE),
        "--fk-function",
        "fk_link6",
        *CHARUCO_ARGS,
        "--output",
        repo_rel(cfg.handeye_output),
    ]


def build_transfer_capture_cmd(cfg: ArmConfig) -> list[str]:
    wrist_flag = "--wrist-topic-left" if cfg.arm == "left" else "--wrist-topic-right"
    joint_flag = "--joint-topic-left" if cfg.arm == "left" else "--joint-topic-right"
    return [
        sys.executable,
        repo_rel(CALIB_DIR / "capture_front_wrist_transfer_samples.py"),
        "--arm",
        cfg.arm,
        "--front-topic",
        FRONT_CAMERA_TOPIC,
        wrist_flag,
        cfg.wrist_topic,
        joint_flag,
        cfg.joint_topic,
        "--front-dir",
        repo_rel(cfg.transfer_front_dir),
        "--wrist-dir",
        repo_rel(cfg.transfer_wrist_dir),
        "--joints-path",
        repo_rel(cfg.transfer_joints_path),
        "--manifest-path",
        repo_rel(cfg.transfer_manifest_path),
        "--prefix",
        cfg.transfer_prefix,
        "--show",
    ]


def build_front_solve_cmd(cfg: ArmConfig) -> list[str]:
    return [
        sys.executable,
        repo_rel(CALIB_DIR / "solve_front_extrinsic_via_wrist.py"),
        "--front-images-dir",
        repo_rel(cfg.transfer_front_dir),
        "--wrist-images-dir",
        repo_rel(cfg.transfer_wrist_dir),
        "--joints",
        repo_rel(cfg.transfer_joints_path),
        "--front-intrinsics",
        repo_rel(CALIB_DATA_DIR / "intrinsics_front_charuco.json"),
        "--wrist-intrinsics",
        repo_rel(cfg.intrinsic_output),
        "--wrist-eye-in-hand",
        repo_rel(cfg.handeye_output),
        "--fk-module",
        repo_rel(FK_MODULE),
        "--fk-function",
        "fk_link6",
        *CHARUCO_ARGS,
        "--output",
        repo_rel(cfg.front_extrinsic_output),
    ]


def run_arm_flow(cfg: ArmConfig, session: WizardSession) -> None:
    print()
    print_success(f"开始执行{cfg.label_cn}完整校准流程。")
    front_intrinsics_path = ensure_front_intrinsics(session)
    print_info(f"后续前置外参求解将使用：{repo_rel(front_intrinsics_path)}")

    capture_step(
        guide_lines=(
            f"步骤 1/6：采集{cfg.label_cn}腕相机内参图片。",
            f"请手持标定板在{cfg.label_cn}镜头前缓慢移动，尽量覆盖画面中心、四角和不同距离。",
            "请保持标定板清晰对焦，尽量避免运动模糊，并多拍几张不同角度的图像。",
        ),
        command=build_intrinsic_capture_cmd(cfg.wrist_topic, cfg.intrinsic_images_dir, cfg.intrinsic_prefix),
        capture_dir=cfg.intrinsic_images_dir,
        capture_label=f"{cfg.label_cn}腕相机内参采集目录",
    )

    solve_step(
        guide_lines=(
            "步骤 2/6：静默解算腕相机内参。",
            "程序将根据刚才采集的图片自动生成相机内参 JSON。",
        ),
        command=build_intrinsic_solve_cmd(cfg.intrinsic_images_dir, cfg.intrinsic_output),
        output_path=cfg.intrinsic_output,
        step_name="腕相机内参解算",
    )

    capture_step(
        guide_lines=(
            "步骤 3/6：采集腕相机 eye-in-hand 样本。",
            "请把标定板固定在桌面上，要求它在整个采集过程中绝对静止，不要挪动桌面上的板。",
            f"然后缓慢移动{cfg.label_cn}和机械臂到多个姿态，确保标定板始终清晰可见。",
        ),
        command=build_handeye_capture_cmd(cfg),
        capture_dir=cfg.handeye_images_dir,
        capture_label=f"{cfg.label_cn} eye-in-hand 采集目录",
    )

    solve_step(
        guide_lines=(
            "步骤 4/6：静默解算 wrist eye-in-hand 外参。",
            "程序将使用腕相机图片、关节角和 FK 模型自动求解手眼外参。",
        ),
        command=build_handeye_solve_cmd(cfg),
        output_path=cfg.handeye_output,
        step_name="wrist eye-in-hand 解算",
    )

    capture_step(
        guide_lines=(
            "步骤 5/6：采集前置-腕部传递样本。",
            f"请让前置相机和{cfg.label_cn}腕相机同时清晰看到同一块标定板。",
            "采集时请保持板面稳定、图像清晰，并尽量让标定板出现在不同位置和朝向。",
        ),
        command=build_transfer_capture_cmd(cfg),
        capture_dir=cfg.transfer_front_dir.parent,
        capture_label=f"{cfg.label_cn}前置-腕部传递采集目录",
    )

    solve_step(
        guide_lines=(
            "步骤 6/6：静默解算前置相机外参。",
            "程序将根据前置/腕部同步样本自动解算 front camera 在当前臂基座下的外参。",
        ),
        command=build_front_solve_cmd(cfg),
        output_path=cfg.front_extrinsic_output,
        step_name="前置外参解算",
    )

    print()
    print_info("步骤 7/7：生成 pipeline 可直接读取的最终文件。")
    final_files = copy_final_outputs(cfg.front_extrinsic_output, cfg.final_targets)
    for path in final_files:
        print_success(f"已生成：{path}")
    print_success("当前这侧的 pipeline 兼容文件已经更新。")
    print_info("如果左右两侧都已完成，就可以直接运行 run_aloha_bimanual_pipeline.sh。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aloha 双臂相机校准一体化交互式向导。")
    parser.parse_args()
    session = WizardSession()

    while True:
        choice = prompt_menu_choice()
        if choice == "3":
            print_success("已退出。")
            return
        cfg = build_arm_config("left" if choice == "1" else "right")
        try:
            run_arm_flow(cfg, session)
        except SystemExit as exc:
            if exc.code in (0, None):
                return
            if exc.code == 130:
                print_warning("收到中断信号，已退出。")
                return
            print_warning("当前流程已中止，返回主菜单。")
            continue
        except KeyboardInterrupt:
            print_warning("\n用户中断，已退出。")
            return
        except Exception as exc:
            print_error(f"流程失败：{exc}")
            print_warning("返回主菜单。")


if __name__ == "__main__":
    main()
