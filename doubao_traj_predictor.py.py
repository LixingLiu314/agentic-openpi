"""
Synchronous Future Trajectory Predictor for Inference

设计思路：
- 与离线预处理一致：在当前帧上标注 EEF **未来**将要到达的位置
- 离线时有 GT 未来帧，所以直接检测未来帧的 EEF；
  推理时没有未来帧，所以让 Doubao API 根据当前图像 + task description
  预测 EEF 当前位置和未来目标位置
- 同步阻塞调用 API，每 N 步更新一次轨迹，中间步复用缓存

数据流:
    每 N 步 (默认6):
    1. 收到新 obs 图像
    2. 同步调用 Doubao API → 检测当前 EEF + 预测未来目标
    3. 更新缓存
    4. draw_future_trajectory 画到当前帧
    5. 发送给 Server 推理

    其余步:
    1. 直接从缓存读取上次预测结果
    2. draw_future_trajectory 画到当前帧
    3. 发送给 Server 推理
"""

import time
import logging
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryCache:
    """轨迹缓存"""
    current_eef: Optional[Tuple[int, int]] = None       # 当前 EEF 像素坐标
    future_points: List[Tuple[int, int]] = field(default_factory=list)  # 未来 EEF 预测位置列表
    timestamp: float = 0.0                               # 预测时的时间戳
    request_id: int = 0                                  # 请求 ID
    is_fresh: bool = False                               # 是否是新预测（还没被使用过）



class ManualTrajectoryPredictor:
    """
    人工标注轨迹预测器：通过 OpenCV 弹窗让用户手动点击标注 EEF 当前位置和未来目标位置。

    操作说明（弹窗标题栏会显示）:
    - 左键点击: 添加一个点（第 1 个点 = 当前 EEF，后续点 = 未来目标）
    - 右键点击: 撤销上一个点
    - 按 Enter/Space: 确认并关闭窗口
    - 按 Esc: 跳过本帧（不标注）

    返回格式与 DoubaoTrajectoryPredictor.predict() 一致：
    - current_eef: 当前 EEF 像素坐标 (x, y)
    - future_points: 未来 EEF 目标位置列表
    """

    # 绘制颜色常量 (BGR)
    CURRENT_COLOR = (255, 0, 0)   # 蓝色: 当前 EEF
    FUTURE_COLOR = (0, 0, 255)    # 红色: 未来目标
    LINE_COLOR = (0, 255, 255)    # 黄色: 连线
    ARROW_COLOR = (0, 255, 0)     # 绿色: 当前步箭头
    POINT_RADIUS = 6
    LINE_THICKNESS = 2

    def __init__(self, window_name: str = "Manual Trajectory Annotation"):
        self.window_name = window_name
        self._points: List[Tuple[int, int]] = []
        self._canvas: Optional[np.ndarray] = None
        self._base_image: Optional[np.ndarray] = None
        self._done = False
        self._cancelled = False

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """OpenCV 鼠标回调：左键添加点，右键撤销。"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._points.append((x, y))
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self._points:
                self._points.pop()
                self._redraw()

    def _redraw(self) -> None:
        """根据当前点列表重绘画布。"""
        if self._base_image is None:
            return
        img = self._base_image.copy()
        h, w = img.shape[:2]

        # 绘制操作提示
        instructions = [
            "LClick: add point (1st=current EEF, rest=future)",
            "RClick: undo | Enter/Space: confirm | Esc: skip",
        ]
        for i, text in enumerate(instructions):
            cv2.putText(img, text, (10, 20 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 绘制状态
        n = len(self._points)
        status = f"Points: {n} (need >=2)" if n < 2 else f"Points: {n} (ready)"
        cv2.putText(img, status, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        if n == 0:
            self._canvas = img
            cv2.imshow(self.window_name, self._canvas)
            return

        # 画连线（从第 2 段开始，第 1 段留给箭头）
        for i in range(1, n - 1):
            cv2.line(img, self._points[i], self._points[i + 1],
                     self.LINE_COLOR, self.LINE_THICKNESS)

        # 画圆点
        for i, pt in enumerate(self._points):
            color = self.CURRENT_COLOR if i == 0 else self.FUTURE_COLOR
            label = "EEF" if i == 0 else f"F{i}"
            cv2.circle(img, pt, self.POINT_RADIUS, color, -1)
            cv2.putText(img, label, (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # 画绿色箭头（当前 EEF → 第一个未来点）
        if n >= 2:
            pt1, pt2 = self._points[0], self._points[1]
            dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
            line_length = np.sqrt(dx**2 + dy**2)
            if line_length > 0:
                tip_ratio = min(10.0 / line_length, 0.5)
                cv2.arrowedLine(img, pt1, pt2, self.ARROW_COLOR,
                                self.LINE_THICKNESS, tipLength=tip_ratio)

        self._canvas = img
        cv2.imshow(self.window_name, self._canvas)

    def predict(
        self, image: np.ndarray, task_description: str = ""
    ) -> Optional[Dict]:
        """
        弹出 OpenCV 窗口让用户手动标注轨迹点。

        Args:
            image: 当前观测图像 (H, W, 3)
            task_description: 当前任务描述（会显示在窗口中）

        Returns:
            {"current_eef": (x, y), "future_points": [(x1, y1), ...]}
            如果用户按 Esc 跳过，返回 None
        """
        self._points = []
        self._done = False
        self._cancelled = False
        # OpenCV 的 imshow/putText 等函数期望 BGR 格式，而流水线中图像为 RGB
        self._base_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 在图像上显示任务描述
        if task_description:
            h_img = self._base_image.shape[0]
            task_text = f"Task: {task_description[:80]}"
            cv2.putText(self._base_image, task_text, (10, h_img - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, max(image.shape[1], 640), max(image.shape[0], 480))
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        # 初始显示
        self._redraw()

        print(f"\n{'='*60}")
        print(f"  [Manual Trajectory Annotation]")
        print(f"  Task: {task_description[:60]}")
        print(f"  Left-click: add point (1st=EEF, rest=future targets)")
        print(f"  Right-click: undo last point")
        print(f"  Enter/Space: confirm | Esc: skip this frame")
        print(f"{'='*60}")

        # 事件循环
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == 27:  # Esc
                self._cancelled = True
                break
            elif key in (13, 32):  # Enter or Space
                if len(self._points) >= 2:
                    self._done = True
                    break
                else:
                    print("  Need at least 2 points (1 current + 1 future). Keep clicking!")

        cv2.destroyWindow(self.window_name)
        cv2.waitKey(1)  # 确保窗口关闭

        if self._cancelled or len(self._points) < 2:
            if self._cancelled:
                print("  Skipped by user (Esc)")
            return None

        current_eef = self._points[0]
        future_points = self._points[1:]
        print(f"  Annotated: current={current_eef}, future={len(future_points)} points")

        return {"current_eef": current_eef, "future_points": future_points}


class DoubaoTrajectoryPredictor:
    """
    Doubao API 封装：预测图像中的 EEF 当前位置 + 未来目标位置。

    返回格式与离线预处理一致：
    - current_eef: 当前 EEF 像素坐标
    - future_points: 未来 EEF 目标位置列表（1~5 个点）
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "doubao-seed-2-0-lite-260215",
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        timeout: float = 50.0,
    ):
        import openai
        import base64

        self.api_key = api_key or self._get_api_key()
        self.model = model
        self.timeout = timeout
        self.base64 = base64

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=base_url,
        )
        self.client.timeout = timeout

    @staticmethod
    def _get_api_key() -> str:
        import os
        key = os.environ.get("VOLCENKEY")
        if not key:
            raise ValueError("VOLCENKEY environment variable not set")
        return key


    def _build_prompt(self, task_description: str) -> str:
        """\n\n"
            "Please follow these steps strictly:\n"
            "1. SCENE ANALYSIS: Briefly describe the spatial relationship between the end-effector, the target object, and any obstacles. Determine the current gripper state (i.e., whether it is empty or holding an object).\n"
            "2. STRATEGY: Explain the movement strategy (e.g., 'move up to avoid the bowl, then translate right to the block').\n"
            "3. WAYPOINTS: Predict the trajectory
        """
        # System Prompt: 负责定义角色、全局规则、坐标系标准和严格的输出格式
        system_prompt = (
            "You are an expert Vision-Language-Action model analyzing robot manipulation scenes. "
            "Your objective is to identify the current end-effector position and predict its future trajectory using 1 to 5 KEYFRAMES (e.g., pre-grasp, grasp, lift, place).\n\n"
            "Return your final coordinates in EXACTLY this format, placed at the very end of your response:\n"
            "<coordinates>\n"
            "<current>x,y</current>\n"
            "<future>x1,y1</future>\n"
            "<future>x2,y2</future>\n"
            "...(more future points if needed)\n"
            "</coordinates>\n\n"
            "Coordinate System Constraints:\n"
            "- All coordinates MUST be normalized to a 0-1000 scale.\n"
            "- (0,0) is top-left, (1000,1000) is bottom-right.\n"
            "- Order future points sequentially."
        )
        # User Prompt: 负责传递当前具体的任务描述（通常在此处也会附带图像输入）
        user_prompt = (
            f"The robot's current task is: \"{task_description}\"\n\n"
            "Please analyze the provided scene and output the waypoints."
        )

        return system_prompt, user_prompt

    def predict(
        self, image: np.ndarray, task_description: str = ""
    ) -> Optional[Dict]:
        import re
        h, w = image.shape[:2]
        # OpenCV imencode 期望 BGR 格式，流水线中图像为 RGB，需先转换
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode('.jpg', bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            return None
        img_base64 = self.base64.b64encode(buffer).decode('utf-8')
        system_prompt, user_prompt = self._build_prompt(task_description)

        try:
            print(f"🚀 [Doubao API] Sending request... (Task: {task_description[:60]}...)")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    # 1. 首先传入 System Prompt，设定全局规则和角色
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    # 2. 然后传入 User Prompt，包含当前的具体任务和图像输入
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ],
                max_tokens=300,
            )
            text = response.choices[0].message.content.strip()

            # === 强制调试打印 ===
            print(f"📩 [Doubao API] Raw Response:\n{'-'*20}\n{text}\n{'-'*20}")
            # ==================

            return self._parse_response(text, w, h)

        except Exception as e:
            # === 打印报错 ===
            print(f"❌ [Doubao API] Error: {e}")
            logger.warning(f"Doubao API call failed: {e}")
            return None

    def _parse_response(self, text: str, img_w: int, img_h: int) -> Optional[Dict]:
        import re
        def norm_to_pixel(x_norm: int, y_norm: int) -> Tuple[int, int]:
            x = max(0, min(img_w - 1, int(x_norm / 1000 * img_w)))
            y = max(0, min(img_h - 1, int(y_norm / 1000 * img_h)))
            return (x, y)

        coord_pattern = r'\s*(\d+)\s*[,，\s]\s*(\d+)\s*'

        # 解析 current
        current_match = re.search(r'<current>' + coord_pattern + r'</current>', text)
        current_eef = None
        if current_match:
            current_eef = norm_to_pixel(int(current_match.group(1)), int(current_match.group(2)))

        # 解析 future
        future_matches = re.findall(r'<future>' + coord_pattern + r'</future>', text)
        future_points = [norm_to_pixel(int(m[0]), int(m[1])) for m in future_matches]

        # 兼容 <point>
        if current_eef is None and not future_points:
            point_matches = re.findall(r'<point>' + coord_pattern + r'</point>', text)
            if point_matches:
                current_eef = norm_to_pixel(int(point_matches[0][0]), int(point_matches[0][1]))
                future_points = [norm_to_pixel(int(m[0]), int(m[1])) for m in point_matches[1:]]

        # === 调试解析结果 ===
        if current_eef is None and not future_points:
            print("⚠️ [Parser] Failed to extract ANY points from response!")
            return None
        else:
            print(f"✅ [Parser] Success! Current: {current_eef}, Future: {len(future_points)} points")

        return {"current_eef": current_eef, "future_points": future_points}

class TrajectoryVisualizer:
    """
    轨迹可视化器，与离线预处理 GTTrajectoryVisualizer.draw_future_trajectory 逻辑一致。

    绘制: current_point (蓝色) → future_points (红色), 绿色箭头指向下一步（当前步），显示在顶层。
    """

    def __init__(
        self,
        point_radius: int = 5,
        line_thickness: int = 2,
        point_color: Tuple[int, int, int] = (0, 0, 255),
        line_color: Tuple[int, int, int] = (0, 255, 255) ,
        arrow_color: Tuple[int, int, int] = (0, 255, 0),
        current_color: Tuple[int, int, int] = (255, 0, 0),
    ):
        self.point_radius = point_radius
        self.line_thickness = line_thickness
        self.point_color = point_color
        self.line_color = line_color
        self.arrow_color = arrow_color
        self.current_color = current_color

    def draw_future_trajectory(
        self,
        image: np.ndarray,
        future_points: List[Tuple[int, int]],
        current_point: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        在图像上绘制未来轨迹。绘制顺序：连线 → 圆点 → 箭头（顶层）。

        绿色箭头：current_point → future_points[0]，表示"当前步"方向，最后绘制确保在顶层。

        Args:
            image: 原始图像 (H, W, 3)
            future_points: 未来 EEF 目标位置列表（从近到远）
            current_point: 当前 EEF 位置（蓝色圆点）

        Returns:
            标注后的图像副本
        """
        # OpenCV 绘图函数的颜色值按 BGR 语义定义，所以先将 RGB 图转为 BGR 绘制
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if not future_points:
            if current_point is not None:
                cv2.circle(img, current_point, self.point_radius, self.current_color, -1)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 构建完整点序列: [current, future_1, future_2, ...]
        all_points = []
        if current_point is not None:
            all_points.append(current_point)
        all_points.extend(future_points)

        if len(all_points) < 2:
            if all_points:
                cv2.circle(img, all_points[0], self.point_radius, self.point_color, -1)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 第一步: 画黄色连线（future_points 之间，跳过第一段，第一段留给箭头）
        for i in range(1, len(all_points) - 1):
            cv2.line(img, all_points[i], all_points[i + 1],
                     self.line_color, self.line_thickness)

        # 第二步: 画圆点
        for i, pt in enumerate(all_points):
            if i == 0 and current_point is not None:
                color = self.current_color  # 蓝色: 当前 EEF
            else:
                color = self.point_color    # 红色: 未来目标
            cv2.circle(img, pt, self.point_radius, color, -1)

        # 第三步: 最后画绿色箭头（current → future[0]，当前步，覆盖在最顶层）
        pt1, pt2 = all_points[0], all_points[1]
        dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
        line_length = np.sqrt(dx**2 + dy**2)
        if line_length > 0:
            tip_ratio = min(10.0 / line_length, 0.5)
            cv2.arrowedLine(img, pt1, pt2, self.arrow_color,
                            self.line_thickness, tipLength=tip_ratio)
        else:
            cv2.line(img, pt1, pt2, self.arrow_color, self.line_thickness)

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class SyncTrajectoryPredictor:
    """
    同步轨迹预测器。

    每 update_interval 步同步阻塞调用 Doubao API 更新轨迹，
    其余步复用上次缓存结果。无后台线程。

    使用方式：
        predictor = SyncTrajectoryPredictor(enable=True, update_interval=6)

        # 推理循环:
        annotated = predictor.step(image, task_description)  # 同步，每 N 步更新
    """

    def __init__(
        self,
        predictor: Optional[DoubaoTrajectoryPredictor] = None,
        visualizer: Optional[TrajectoryVisualizer] = None,
        update_interval: int = 6,
        enable: bool = True,
        # 保留旧参数名以兼容，但不再使用
        min_request_interval: float = 0.5,
        # --- 新增: 轨迹预测模式 ---
        trajectory_mode: str = "doubao",
    ):
        """
        Args:
            predictor: 轨迹预测器实例（默认根据 trajectory_mode 自动创建）
            visualizer: 轨迹可视化器（默认创建 TrajectoryVisualizer）
            update_interval: 每隔多少步同步调用一次 API 更新轨迹（默认 6）
            enable: 是否启用（False 时所有操作为空操作）
            trajectory_mode: 轨迹预测模式，可选值:
                - "doubao": 使用 Doubao API 自动预测（默认）
                - "manual": 人工标注模式，弹出 OpenCV 窗口用鼠标点击标注
                - "disabled": 禁用轨迹预测（等价于 enable=False）
        """
        # 处理 mode=disabled
        if trajectory_mode == "disabled":
            enable = False
        self.trajectory_mode = trajectory_mode

        self.enable = enable
        if not enable:
            return

        # 根据 mode 选择 predictor
        if predictor is not None:
            self.predictor = predictor
        elif trajectory_mode == "manual":
            self.predictor = ManualTrajectoryPredictor()
            # manual 模式下每步都弹窗（但受 update_interval 控制，缓存步不弹窗）
            print("🖱️  [SyncPredictor] Manual annotation mode enabled")
        else:
            self.predictor = DoubaoTrajectoryPredictor()
        self.visualizer = visualizer or TrajectoryVisualizer()
        self.update_interval = update_interval

        # --- 缓存 ---
        self._cache = TrajectoryCache()
        self._step_counter = 0  # 当前 episode 内的步数计数器

        # --- 统计 ---
        self._stats = {
            "total_requests": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "cache_hits": 0,
            "avg_latency": 0.0,
        }

    def start(self):
        """兼容旧接口，同步模式下为空操作。"""
        if not self.enable:
            return
        logger.info("SyncTrajectoryPredictor ready (no background thread)")

    def stop(self):
        """兼容旧接口，打印统计信息。"""
        if not self.enable:
            return
        logger.info(
            f"SyncTrajectoryPredictor stopped. Stats: "
            f"requests={self._stats['total_requests']}, "
            f"success={self._stats['successful_predictions']}, "
            f"failed={self._stats['failed_predictions']}, "
            f"cache_hits={self._stats['cache_hits']}, "
            f"avg_latency={self._stats['avg_latency']:.2f}s"
        )
        print(
            f"📊 [SyncPredictor] Final Stats: "
            f"requests={self._stats['total_requests']}, "
            f"success={self._stats['successful_predictions']}, "
            f"failed={self._stats['failed_predictions']}, "
            f"cache_hits={self._stats['cache_hits']}, "
            f"avg_latency={self._stats['avg_latency']:.2f}s"
        )

    def reset(self):
        """重置缓存和计数器（episode 切换时调用）。"""
        if not self.enable:
            return
        self._cache = TrajectoryCache()
        self._step_counter = 0
        print("🔄 [SyncPredictor] Reset for new episode")

    def step(
        self, image: np.ndarray, task_description: str = ""
    ) -> np.ndarray:
        """
        同步轨迹预测 + 可视化的统一入口。

        每 update_interval 步同步调用 API 更新轨迹缓存；
        其余步直接复用缓存，将轨迹画到当前帧上返回。

        Args:
            image: 当前观测图像 (H, W, 3)
            task_description: 当前任务描述

        Returns:
            标注了轨迹的图像（如果有缓存），否则返回原图
        """
        if not self.enable:
            return image

        # 判断是否需要更新轨迹
        need_update = (self._step_counter % self.update_interval == 0)
        self._step_counter += 1

        if need_update:
            self._sync_predict(image, task_description)
        else:
            self._stats["cache_hits"] += 1
            next_update = (self._step_counter // self.update_interval + 1) * self.update_interval
            print(
                f"📦 [SyncPredictor] Step {self._step_counter}: "
                f"Using cached trajectory (next update at step {next_update})"
            )

        # 从缓存读取并绘制
        return self._draw_from_cache(image)

    # ─── 兼容旧接口（保留 submit_image + get_trajectory_image） ───

    def submit_image(self, image: np.ndarray, task_description: str = ""):
        """兼容旧接口。同步模式下仅暂存图像和任务，不立即调用 API。"""
        if not self.enable:
            return
        self._pending_image = image.copy()
        self._pending_task = task_description

    def get_trajectory_image(
        self, image: np.ndarray, block: bool = False, timeout: float = 10.0
    ) -> np.ndarray:
        """兼容旧接口。同步模式下等价于调用 step()。"""
        if not self.enable:
            return image
        task = getattr(self, '_pending_task', '')
        return self.step(image, task)

    # ─── 内部方法 ───

    def _sync_predict(self, image: np.ndarray, task_description: str):
        """同步阻塞调用 API 更新缓存。"""
        self._stats["total_requests"] += 1
        start_time = time.time()

        print(f"🔄 [SyncPredictor] Step {self._step_counter}: Calling API (blocking)...")
        result = self.predictor.predict(image, task_description)

        latency = time.time() - start_time

        # 更新平均延迟
        n = self._stats["total_requests"]
        self._stats["avg_latency"] = (
            self._stats["avg_latency"] * (n - 1) + latency
        ) / n

        if result is not None:
            self._stats["successful_predictions"] += 1
            self._cache.current_eef = result["current_eef"]
            self._cache.future_points = result["future_points"]
            self._cache.timestamp = time.time()
            self._cache.is_fresh = True
            self._cache.request_id += 1
            print(
                f"✅ [SyncPredictor] Prediction done: current={result['current_eef']}, "
                f"future={len(result['future_points'])} points, latency={latency:.2f}s"
            )
        else:
            self._stats["failed_predictions"] += 1
            print(f"❌ [SyncPredictor] Prediction failed, latency={latency:.2f}s")

    def _draw_from_cache(self, image: np.ndarray) -> np.ndarray:
        """从缓存读取轨迹点并绘制到图像上。"""
        current_eef = self._cache.current_eef
        future_points = list(self._cache.future_points)
        self._cache.is_fresh = False

        if current_eef is None and not future_points:
            print("⚪ [SyncPredictor] No trajectory in cache. Returning raw image.")
            return image

        print(f"🎨 [SyncPredictor] Drawing trajectory: current={current_eef}, {len(future_points)} future points")
        return self.visualizer.draw_future_trajectory(
            image,
            future_points=future_points,
            current_point=current_eef,
        )

    def get_cached_prediction(self) -> Optional[Dict]:
        """获取最新缓存的预测结果。"""
        if not self.enable:
            return None
        if self._cache.current_eef is None and not self._cache.future_points:
            return None
        return {
            "current_eef": self._cache.current_eef,
            "future_points": list(self._cache.future_points),
            "timestamp": self._cache.timestamp,
        }

    @property
    def stats(self) -> dict:
        return self._stats.copy()


# 保留旧名称作为别名，确保向后兼容
AsyncTrajectoryPredictor = SyncTrajectoryPredictor