"""
EvalCoTManager
==============
Wraps ModelClient (the HTTP policy client) with online CoT generation
during SimplerEnv evaluation.

Two-phase CoT strategy
-----------------------
Phase-1  (window_id == 0, step=0):
    Call doubao with image_0 + task description.
    Output: all_subtasks (list) + current_subtask (str).
    Result is immediately active for window 0.
    image_0 is cached as prev_window_image.

Phase-2  (window_id > 0, at each window boundary):
    Call doubao with prev_window_image + all_subtasks + task description.
    Output: current_subtask (str), immediately active for this window.
    Current image is then cached as new prev_window_image.

Window image used at boundary N  →  image from the START of window N-1.

Integration (start_simpler_env.py)
-----------------------------------
    # Before:
    model = ModelClient(...)

    # After:
    from examples.SimplerEnv.eval_files.eval_cot_manager import DoubaoVLM, EvalCoTManager
    base_model = ModelClient(...)
    vlm = DoubaoVLM(api_key="YOUR_KEY")
    model = EvalCoTManager(base_model, vlm, cot_refresh_interval=6)

EvalCoTManager exposes the same reset() / step() interface as ModelClient.
The task_description must be passed to reset() so the manager can use it
when triggering VLM calls during subsequent steps.
"""

import base64
import io
import os
import re
import time
from typing import List, Optional

import numpy as np
import openai
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────
######################original########################
PHASE1_SYSTEM_PROMPT = """\
You are a robotic manipulation assistant.
Given the robot's initial camera view and task instruction, produce:
  1. An ordered list of ALL high-level subtasks to complete the task.
  2. The subtask the robot should focus on RIGHT NOW (always the first one at the start).

Rules for subtask list:
- Each subtask is one concise sentence (a meaningful manipulation stage).
- Use exact object names from the task instruction.
- Spatial descriptions (left/right) are from the robot's viewpoint.
- No low-level primitives (e.g., "open gripper to 30 mm").

Example 1: 
Task: put eggplant into yellow basket
All subtasks:
1. Reach for the eggplant in the sink.
2. Grasp the eggplant in the sink.
3. Lift and move the eggplant towards the yellow basket.
4. Position the eggplant above the yellow basket.
5. Release the eggplant into the yellow basket.

Task: put spoon on the table cloth
All subtasks:
1. Reach for the spoon until the end effector is positioned on either side of the spoon.
2. Grasp the spoon.
3. Lift and move towards the spoon.
4. Position the spoon above the table cloth.
5. Release the spoon onto the table cloth.

Task: stack green cube on yellow cube
All subtasks:
1. Reach for the green cube until the end effector is positioned on either side of the green cube.
2. Grasp the green cube.
3. Lift and move towards the green cube.
4. Position the green cube above the yellow cube.
5. Release the green cube onto the yellow cube.

Output format — follow EXACTLY, no extra text:
All subtasks:
1. <subtask>
2. <subtask>
...

Current subtask: <copy the first subtask verbatim>
"""


########################################### combine the original and 与仿真中的subtask相近的版本################
# PHASE1_SYSTEM_PROMPT = """\
# You are a robotic manipulation assistant.
# Given the robot's initial camera view and task instruction, produce:
#   1. An ordered list of ALL high-level subtasks to complete the task.
#   2. The subtask the robot should focus on RIGHT NOW (always the first one at the start).

# Rules for subtask list:
# - Each subtask is one concise sentence (a meaningful manipulation stage).
# - Use exact object names from the task instruction.
# - Spatial descriptions (left/right) are from the robot's viewpoint.
# - No low-level primitives (e.g., "open gripper to 30 mm").

# Example 1: 
# Task: put spoon on the table cloth
# All subtasks:
# 1. Reach for the spoon until the end effector is positioned on either side of the spoon.
# 2. Grasp the spoon.
# 3. Lift and move towards the spoon.
# 4. Position the spoon above the table cloth.
# 5. Release the spoon onto the table cloth.

# Example 2:
# Task: put carrot on plate
# All subtasks:
# 1. Reach for the carrot
# 2. Grasp the carrot
# 3. Lift the carrot
# 4. Move carrot to the plate
# 5. Place carrot on the plate
# 6. Release the carrot

# Example 3:
# Task: stack green cube on yellow cube
# All subtasks:
# 1. Reach for the green cube until the end effector is positioned on either side of the green cube.
# 2. Grasp the green cube.
# 3. Lift and move towards the green cube.
# 4. Position the green cube above the yellow cube.
# 5. Release the green cube onto the yellow cube.

# Example 4: 
# Task: put eggplant into yellow basket
# All subtasks:
# 1. Approach the eggplant
# 2. Grasp the eggplant
# 3. Lift the eggplant
# 4. Move the eggplant towards the yellow basket
# 5. Release the eggplant into the yellow basket
# 6. Move away from the yellow basket

# Output format — follow EXACTLY, no extra text:
# All subtasks:
# 1. <subtask>
# 2. <subtask>
# ...

# Current subtask: <copy the first subtask verbatim>
# """
# #####################################







# ################修改为与训练过程中看到的subtask格式更接近的版本#####################
# PHASE1_SYSTEM_PROMPT = """\
# You are a robotic manipulation assistant.
# Given the robot's initial camera view and task instruction, produce:
#   1. An ordered list of ALL high-level subtasks to complete the task.
#   2. The subtask the robot should focus on RIGHT NOW (always the first one at the start).
 
# Rules for subtask list:
# - Each subtask is one concise phrase (typically 3–8 words, a meaningful manipulation stage).
# - Use exact object names from the task instruction.
# - The number of subtasks should match the task complexity (typically 4–7).
# - No low-level primitives (e.g., "open gripper to 30 mm").
 
# Example 1: 
# Task: put the spoon on the towel
# All subtasks:
# 1. Reach for the spoon
# 2. Grasp the spoon
# 3. Lift the spoon
# 4. Move the spoon towards the towel
# 5. Release the spoon onto the towel
# 6. Move away from the spoon
 
# Example 2:
# Task: put carrot on plate
# All subtasks:
# 1. Reach for the carrot
# 2. Grasp the carrot
# 3. Lift the carrot
# 4. Move carrot to the plate
# 5. Place carrot on the plate
# 6. Release the carrot
 
# Example 3:
# Task: stack the green block on the yellow block
# All subtasks:
# 1. Reach for the green block
# 2. Grasp the green block
# 3. Lift the green block
# 4. Move the green block above the yellow block
# 5. Release the green block
# 6. Move away from the green block
 
# Example 4: 
# Task: put eggplant into yellow basket
# All subtasks:
# 1. Approach the eggplant
# 2. Grasp the eggplant
# 3. Lift the eggplant
# 4. Move the eggplant towards the yellow basket
# 5. Release the eggplant into the yellow basket
# 6. Move away from the yellow basket
 
# Output format — follow EXACTLY, no extra text:
# All subtasks:
# 1. <subtask>
# 2. <subtask>
# ...
 
# Current subtask: <copy the first subtask verbatim>
# """
 #####################################





##############################################只修改spoon
# PHASE1_SYSTEM_PROMPT = """\
# You are a robotic manipulation assistant.
# Given the robot's initial camera view and task instruction, produce:
#   1. An ordered list of ALL high-level subtasks to complete the task.
#   2. The subtask the robot should focus on RIGHT NOW (always the first one at the start).

# Rules for subtask list:
# - Each subtask is one concise sentence (a meaningful manipulation stage).
# - Use exact object names from the task instruction.
# - Spatial descriptions (left/right) are from the robot's viewpoint.
# - No low-level primitives (e.g., "open gripper to 30 mm").

# Example 1: 
# Task: put eggplant into yellow basket
# All subtasks:
# 1. Reach for the eggplant in the sink.
# 2. Grasp the eggplant in the sink.
# 3. Lift and move the eggplant towards the yellow basket.
# 4. Position the eggplant above the yellow basket.
# 5. Release the eggplant into the yellow basket.

# Example 2: 
# Task: put the spoon on the towel
# All subtasks:
# 1. Reach for the spoon until the end effector is positioned on either side of the green spoon.
# 2. Grasp the spoon.
# 3. Lift the spoon.
# 4. Move the spoon towards the towel.

# Example 3:
# Task: stack green cube on yellow cube
# All subtasks:
# 1. Reach for the green cube until the end effector is positioned on either side of the green cube.
# 2. Grasp the green cube.
# 3. Lift and move towards the green cube.
# 4. Position the green cube above the yellow cube.
# 5. Release the green cube onto the yellow cube.

# Output format — follow EXACTLY, no extra text:
# All subtasks:
# 1. <subtask>
# 2. <subtask>
# ...

# Current subtask: <copy the first subtask verbatim>
# """








PHASE1_USER_PROMPT = """\
Task: {task_description}
The initial camera observation is provided in the image.
"""

PHASE2_SYSTEM_PROMPT = """\
You are a robotic manipulation assistant.
Given the robot's camera view from the previous observation window, a task description,
and the full ordered subtask list, identify which subtask the robot should focus on NOW.

Rules:
- Assess the image carefully (gripper state, object positions, task progress).
- Output ONLY the text of the single most relevant subtask — copy it verbatim from the list.
- Do NOT output a number, explanation, or anything else.
"""

PHASE2_USER_PROMPT = """\
Task: {task_description}

All subtasks (ordered):
{all_subtasks_str}

The camera observation from the previous window is provided in the image.

Which subtask should the robot focus on now? Output only the subtask text:
"""


# ──────────────────────────────────────────────────────────────────────────────
# Doubao VLM client
# ──────────────────────────────────────────────────────────────────────────────

class DoubaoVLM:
    """Thin wrapper around the doubao / OpenAI-compatible vision-language API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        model: str = "doubao-seed-2-0-pro-260215",
        log_dir: str = "./cot_logs",   # ← 新增：日志保存目录
    ):
        for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy"):
            os.environ.pop(var, None)
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.log_dir = log_dir
        self._call_counter = 0   # 全局计数，保证文件名不冲突
        os.makedirs(log_dir, exist_ok=True)
        

    def _save_log(self, tag: str, image, text_output: str, extra_info: str = ""):
        """保存图片 + 文本到 log_dir，文件名含计数和 tag。"""
        idx = self._call_counter
        self._call_counter += 1

        # 保存图片
        pil = self._to_pil(image)
        img_path = os.path.join(self.log_dir, f"{idx:04d}_{tag}.jpg")
        pil.save(img_path, format="JPEG", quality=90)

        # 保存文本
        txt_path = os.path.join(self.log_dir, f"{idx:04d}_{tag}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            if extra_info:
                f.write(f"[Info]\n{extra_info}\n\n")
            f.write(f"[Doubao Output]\n{text_output}\n")

        print(f"[DoubaoVLM] Log saved → {img_path}")

    # ── image utils ───────────────────────────────────────────────────────────

    @staticmethod
    def _to_pil(image) -> Image.Image:
        """Accept PIL / numpy / torch Tensor → RGB PIL Image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(image).convert("RGB")
        try:
            import torch
            if isinstance(image, torch.Tensor):
                arr = image.cpu().numpy()
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                    arr = arr.transpose(1, 2, 0)  # CHW → HWC
                if arr.dtype != np.uint8:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                return Image.fromarray(arr).convert("RGB")
        except ImportError:
            pass
        raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _encode_pil(pil: Image.Image) -> str:
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _call(self, user_text: str, image, system_text: Optional[str] = None) -> str:
        pil = self._to_pil(image)
        b64 = self._encode_pil(pil)
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": user_text},
        ]
        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": content})

        import time
        for attempt in range(6):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if "429" in str(e) or "RateLimit" in type(e).__name__ or "TooManyRequests" in str(e):
                    wait = 2 ** attempt  # 1, 2, 4, 8, 16, 32 seconds
                    print(f"[RateLimit] attempt {attempt+1}/6, retrying in {wait}s ...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("VLM _call failed after 6 retries due to rate limiting")

    # ── public API ────────────────────────────────────────────────────────────

    def phase1(self, task_description: str, image) -> dict:
        """
        Episode start: generate all_subtasks + current_subtask from image_0.

        Returns
        -------
        dict:
            "all_subtasks"    : List[str]
            "current_subtask" : str
        """
        user_prompt = PHASE1_USER_PROMPT.format(task_description=task_description)
        t0 = time.time()
        raw = self._call(user_prompt, image, system_text=PHASE1_SYSTEM_PROMPT)
        print(f"[DoubaoVLM] Phase-1 ({time.time()-t0:.2f}s):\n{raw}\n")

        # ── 保存日志 ──
        self._save_log(
            tag="phase1",
            image=image,
            text_output=raw,
            extra_info=f"task: {task_description}",
        )
        # ─────────────

        all_subtasks: List[str] = []
        current_subtask: str = ""
        in_list = False

        for line in raw.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.lower().startswith("all subtasks"):
                in_list = True
                continue
            if s.lower().startswith("current subtask"):
                in_list = False
                parts = s.split(":", 1)
                if len(parts) == 2:
                    current_subtask = parts[1].strip()
                continue
            if in_list:
                cleaned = re.sub(r"^[\d]+[.)]\s*", "", s).strip()
                cleaned = re.sub(r"^[-•]\s*", "", cleaned).strip()
                if cleaned:
                    all_subtasks.append(cleaned)

        # Fallbacks
        if not all_subtasks:
            all_subtasks = [raw]
        if not current_subtask and all_subtasks:
            current_subtask = all_subtasks[0]

        return {"all_subtasks": all_subtasks, "current_subtask": current_subtask}

    def phase2(
        self,
        task_description: str,
        all_subtasks: List[str],
        prev_window_image,
    ) -> str:
        """
        Window boundary: select current subtask using the PREVIOUS window's image.

        Returns the selected subtask string.
        """
        subtasks_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(all_subtasks))
        user_prompt = PHASE2_USER_PROMPT.format(
            task_description=task_description,
            all_subtasks_str=subtasks_str,
        )
        t0 = time.time()
        raw = self._call(user_prompt, prev_window_image, system_text=PHASE2_SYSTEM_PROMPT)
        print(f"[DoubaoVLM] Phase-2 ({time.time()-t0:.2f}s): {raw!r}\n")
        # ── 保存日志 ──
        self._save_log(
            tag="phase2",
            image=prev_window_image,
            text_output=raw,
            extra_info=f"task: {task_description}\nall_subtasks:\n{subtasks_str}",
        )
        # ─────────────
        cleaned = re.sub(r"^[\d]+[.)]\s*", "", raw.strip()).strip()
        return cleaned

class HumanVLM:
    """
    用于调试：用人工输入替代 DoubaoVLM 的调用。
    每次触发时显示当前图片，然后在终端等待输入。
    """

    def __init__(self, log_dir: str = "./cot_logs_human"):
        self.log_dir = log_dir
        self._call_counter = 0
        self.preview_path = "./cot_current_obs.jpg"
        os.makedirs(log_dir, exist_ok=True)

    @staticmethod
    def _to_pil(image) -> Image.Image:
        """同 DoubaoVLM._to_pil"""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(image).convert("RGB")
        try:
            import torch
            if isinstance(image, torch.Tensor):
                arr = image.cpu().numpy()
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                    arr = arr.transpose(1, 2, 0)
                if arr.dtype != np.uint8:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                return Image.fromarray(arr).convert("RGB")
        except ImportError:
            pass
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _show_and_save(self, image, tag: str) -> str:
        """显示图片并保存，返回保存路径。"""
        pil = self._to_pil(image)
    
        # 带计数的归档路径
        img_path = os.path.join(self.log_dir, f"{self._call_counter:04d}_{tag}.jpg")
        pil.save(img_path, format="JPEG", quality=90)
        self._call_counter += 1

        # 固定预览路径，每次覆盖，方便直接查看
        pil.save(self.preview_path, format="JPEG", quality=90)

        print(f"[HumanVLM] 当前图片 → {self.preview_path}  (归档: {img_path})")
        return img_path

    def phase1(self, task_description: str, image) -> dict:
        """
        Phase-1：显示 image_0，人工输入 all_subtasks + current_subtask。

        输入格式：
          第一步：逐行输入子任务，输入空行结束。
          第二步：输入 current_subtask（直接回车则默认取第一条）。
        """
        self._show_and_save(image, tag="phase1_step0")

        print("\n" + "="*60)
        print(f"[Phase-1] Task: {task_description}")
        print("请逐行输入 all_subtasks（输入空行结束）：")

        all_subtasks: List[str] = []
        while True:
            line = input(f"  subtask {len(all_subtasks)+1}: ").strip()
            if not line:
                break
            all_subtasks.append(line)

        if not all_subtasks:
            all_subtasks = ["(no subtask provided)"]

        print(f"\n已输入 {len(all_subtasks)} 条子任务：")
        for i, s in enumerate(all_subtasks):
            print(f"  {i+1}. {s}")

        default = all_subtasks[0]
        current_subtask = input(
            f"\n请输入 current_subtask（直接回车默认使用第1条）: "
        ).strip()
        if not current_subtask:
            current_subtask = default

        print(f"[Phase-1] current_subtask = {current_subtask!r}")
        print("="*60 + "\n")

        # 保存文本记录
        txt_path = os.path.join(self.log_dir, f"{self._call_counter-1:04d}_phase1.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"task: {task_description}\n\nall_subtasks:\n")
            for i, s in enumerate(all_subtasks):
                f.write(f"{i+1}. {s}\n")
            f.write(f"\ncurrent_subtask: {current_subtask}\n")

        return {"all_subtasks": all_subtasks, "current_subtask": current_subtask}

    def phase2(
        self,
        task_description: str,
        all_subtasks: List[str],
        prev_window_image,
    ) -> str:
        """
        Phase-2：显示 prev_window_image，人工输入 current_subtask。
        """
        self._show_and_save(prev_window_image, tag="phase2")

        print("\n" + "="*60)
        print(f"[Phase-2] Task: {task_description}")
        print("all_subtasks（供参考）：")
        for i, s in enumerate(all_subtasks):
            print(f"  {i+1}. {s}")

        current_subtask = input("\n请输入 current_subtask: ").strip()
        if not current_subtask:
            current_subtask = all_subtasks[0]
            print(f"  （空输入，默认使用第1条: {current_subtask!r}）")

        print(f"[Phase-2] current_subtask = {current_subtask!r}")
        print("="*60 + "\n")

        # 保存文本记录
        txt_path = os.path.join(self.log_dir, f"{self._call_counter-1:04d}_phase2.txt")
        subtasks_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(all_subtasks))
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"task: {task_description}\n\nall_subtasks:\n{subtasks_str}\n")
            f.write(f"\ncurrent_subtask: {current_subtask}\n")

        return current_subtask
# ──────────────────────────────────────────────────────────────────────────────
# EvalCoTManager
# ──────────────────────────────────────────────────────────────────────────────

class EvalCoTManager:
    """
    Drop-in wrapper around ModelClient that injects CoT metadata into every
    call to step().

    State machine per episode
    -------------------------
    step=0,  window=0:  Phase-1(image_0) → all_subtasks, active_subtask
                        prev_image ← image_0

    step=K*interval, window=K>0:
                        Phase-2(prev_image) → active_subtask
                        prev_image ← image_K*interval

    Between boundaries: active_subtask unchanged.
    """

    def __init__(
        self,
        base_model,
        vlm: DoubaoVLM,
        cot_refresh_interval: int = 6,
    ):
        self.base_model = base_model
        self.vlm = vlm
        self.cot_refresh_interval = cot_refresh_interval
        self._vlm_root_log_dir = vlm.log_dir 
        self._episode_counter = 0
        self._reset_state()

    def _reset_state(self):
        self._step_index: int = 0
        self._prev_window_id: int = -1
        self._prev_window_image = None
        self._all_subtasks: Optional[List[str]] = None
        self._active_subtask: Optional[str] = None
        self._task_description: str = ""

    @staticmethod
    def _copy_image(image):
        if isinstance(image, Image.Image):
            return image.copy()
        if isinstance(image, np.ndarray):
            return image.copy()
        return image  # torch Tensor: assume caller manages lifetime

    def _maybe_update_cot(self, current_image):
        window_id = self._step_index // self.cot_refresh_interval

        if window_id == self._prev_window_id:
            return  # still inside the same window

        self._prev_window_id = window_id

        if window_id == 0:
            # ── Phase-1 ──────────────────────────────────────────────────────
            print(f"[EvalCoTManager] step={self._step_index} | window=0 → Phase-1")
            result = self.vlm.phase1(self._task_description, current_image)
            self._all_subtasks   = result["all_subtasks"]
            self._active_subtask = result["current_subtask"]
            # Cache image_0 so Phase-2 at window=1 can use it
            self._prev_window_image = self._copy_image(current_image)

        else:
            # ── Phase-2 ──────────────────────────────────────────────────────
            print(f"[EvalCoTManager] step={self._step_index} | window={window_id} → Phase-2")
            if self._all_subtasks and self._prev_window_image is not None:
                self._active_subtask = self.vlm.phase2(
                    self._task_description,
                    self._all_subtasks,
                    self._prev_window_image,  # ← previous window's image
                )
            else:
                print("[EvalCoTManager] Warning: missing all_subtasks or prev_window_image, skipping")

            # Cache current image for the NEXT window boundary
            self._prev_window_image = self._copy_image(current_image)

    # ── ModelClient interface ─────────────────────────────────────────────────

    def reset(self, task_description: str = "",  **kwargs):
        # evaluator 调用的是 model.reset(task_description)，位置参数
        self._reset_state()
        self._task_description = task_description
        # ── 每个 episode 用独立子目录 ──
        import time as _time
        task_slug = task_description[:40].replace(' ', '_')
        episode_tag = f"{self._episode_counter:03d}"
        self._episode_counter += 1
        episode_log_dir = os.path.join(
            self._vlm_root_log_dir,
            task_slug,
            f"ep_{episode_tag}"
        )
        os.makedirs(episode_log_dir, exist_ok=True)
        self.vlm.log_dir = episode_log_dir
        self.vlm._call_counter = 0
        # ──────────────────────────────

        print(f"[EvalCoTManager] Episode reset. Task: {task_description!r}")
        return self.base_model.reset(task_description, **kwargs)

    def step(self, image, state=None, task_description: Optional[str] = None,  **kwargs):
        # Supports both ModelClient-style step(image, task_description) and
        # Bridge eval step(image, state, task_description).
        if isinstance(state, str) and task_description is None:
            task_description = state
            state = None
        if task_description is not None:
            self._task_description = task_description

        self._maybe_update_cot(image)

        cot_metadata = {}
        if self._all_subtasks is not None:
            cot_metadata["all_subtasks"] = self._all_subtasks
        if self._active_subtask is not None:
            cot_metadata["current_subtask"] = self._active_subtask

        # 透传 task_description，同时附带 cot_metadata
        if state is None:
            result = self.base_model.step(
                image,
                task_description,
                cot_metadata=cot_metadata if cot_metadata else None,
                **kwargs,
            )
        else:
            result = self.base_model.step(
                image,
                state,
                task_description,
                cot_metadata=cot_metadata if cot_metadata else None,
                **kwargs,
            )

        self._step_index += 1
        return result
    def __getattr__(self, name):
        """Delegate any other attribute access to the wrapped ModelClient."""
        return getattr(self.base_model, name)
