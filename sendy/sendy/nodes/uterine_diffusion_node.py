#!/usr/bin/env python3
"""
UterineDiffusionNode

Same model + same inference pipeline as diffusion_displacement's push_node.py /
diffusion_node.py (DisplacementDiffusionSDE, VP-SDE, image + start-point
conditioning, 10 waypoints x 3 = 30-dim displacement action).

Driven by the manual state machine: subscribes to /current_task (published by
task_publisher_node.py) and, whenever the task name matches one of the 3
diffusion-driven tasks below, runs inference and publishes the resulting
trajectory. "screw" is NOT handled here — it's not model-driven, see
uterine_move_node.py.

    retract_start   left arm    approach trajectory before the screw step
    retract         left arm    main retraction trajectory
    resect_start    right arm   approach trajectory before cautery
    resect          right arm   main cautery/resection trajectory

Hardware interface is taken from retract_fib_diffusion_node.py / move_node.py:
  - camera:            /image             (sensor_msgs/Image, shared by both arms)
  - left start point:  smoother_uterus/left/tip in hy/left/fwkin   (TF from smoother_uterus)
  - right start point: smoother_uterus/right/tip in hy/right/fwkin (TF from smoother_uterus)
  - output: one PointCloud2 topic per task (see ACTION_TOPICS below), each
    carrying the FULL predicted trajectory (N_WAYPOINTS points), consumed by
    uterine_move_node.py (left arm) / uterine_move_node_r.py (right arm).

Any task whose weights file is missing is skipped at startup (logged as a
warning) rather than crashing the whole node — useful while only some of the
four models are trained.

No slip-boost retry loop here (yet) — the uterine setup doesn't currently
publish a /done_retract completion signal or have a trained slip classifier
for either arm, unlike the old RetractFibDiffusion pipeline. Flagged as
future work.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image as PILimg

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, TransformException
from ament_index_python.packages import get_package_share_directory

PACKAGE_PATH = get_package_share_directory('sendy')
CUR_PATH = os.path.dirname(__file__)

# Proper relative imports (NOT sys.path.append + top-level import) -- required
# because displacement_diffusion_sde.py itself does `from .diffusion_utils
# import ...`, which only resolves if it's loaded as part of the
# sendy.models.diffusion_displacement package. Loading it as a bare
# top-level module (via sys.path hacking) strips that package context and
# breaks its own internal relative import.
from ..point_cloud_utils.point_cloud_utils import xyz_array_to_pointcloud2
from ..models.diffusion_displacement.displacement_diffusion_sde import DiffusionSDEConfig, DisplacementDiffusionSDE


# ============================================================================
# CONFIG (edit here, not argparse)
# ============================================================================

# NOTE: "screw" is intentionally excluded here — it's handled directly by
# uterine_move_node.py and involves no model inference.
TASK_NAMES = ("retract_start", "retract", "resect_start", "resect")

ARM_FOR_TASK = {
    "retract_start": "left",
    "retract":       "left",
    "resect_start":  "right",
    "resect":        "right",
}

WEIGHT_FILES = {
    "retract_start": "retract_s.pt",
    "retract":       "retract.pt",
    "resect_start":  "resect_s.pt",
    "resect":        "resect.pt",
}

ACTION_TOPICS = {
    "retract_start": "/retract_start_action",
    "retract":       "/retract_action",
    "resect_start":  "/resect_start_action",
    "resect":        "/resect_action",
}

# Start points are the smoother's tracked tips (launch smoother_uterus with left_tool_length:=0.0 and
# right_tool_length:=0.0 so these are the inner-tube ends, the point the models were trained on).
TIP_FRAMES = {
    "left":  "smoother_uterus/left/tip",
    "right": "smoother_uterus/right/tip",
}
MAX_TIP_AGE_S = 1.0                 # refuse to plan from an older tip (smoother not running)

CURRENT_TASK_TOPIC = "/current_task"
CAMERA_TOPIC = "/image"   # shared endoscope view for both arms

IMAGE_SIZE = (224, 224)
N_WAYPOINTS = 10                    # matches action_dim=30 (10 x 3)

# Endoscope FOV crop — MUST match convert_fibroid_data.py (the script that
# produced the training pickles for these checkpoints) or inference sees a
# different image distribution than training did. Auto-detects the circular
# FOV bounding box from non-black pixels, squares + pads it, then resizes.
CROP_BLACK_THRESH = 15   # pixel intensity <= this counts as "black" border
CROP_PAD = 2             # px padding kept around the detected FOV box
CROP_SQUARE = True       # pad the tighter dimension so the crop is square

N_SAMPLES = 5                        # candidate trajectories per inference call
INFER_STEPS = 250
GUIDANCE_SCALE = 0.5
SAMPLER = "sde"                     # "sde" or "ode"

INTERACTIVE_SELECT = True           # pop up matplotlib trajectory picker

# Each arm's start point and trajectory are in that arm's model frame. smoother_uterus's launch publishes
# it as a fixed rotation of smoother_uterus/<side>/base (the axes motor_node's FK uses), so TF and the
# visual servo can use it.
FRAME_IDS = {
    "left":  "hy/left/fwkin",
    "right": "hy/right/fwkin",
}


# ============================================================================
# Image preprocessing (MUST match convert_fibroid_data.py training pipeline)
# ============================================================================

def compute_crop_box(image: np.ndarray, black_thresh: int = CROP_BLACK_THRESH,
                      pad: int = CROP_PAD, square: bool = CROP_SQUARE):
    """
    Detect the bounding box of the non-black endoscope FOV in a single frame.
    Returns (rmin, rmax, cmin, cmax), inclusive. Ported verbatim from
    convert_fibroid_data.py so inference sees the same crop training did.
    """
    h, w = image.shape[:2]
    gray = image.mean(axis=2) if image.ndim == 3 else image
    mask = gray > black_thresh

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return 0, h - 1, 0, w - 1

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    rmin = max(0, rmin - pad)
    rmax = min(h - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(w - 1, cmax + pad)

    if square:
        box_h = rmax - rmin + 1
        box_w = cmax - cmin + 1
        side = max(box_h, box_w)
        rc, cc = (rmin + rmax) // 2, (cmin + cmax) // 2
        rmin = max(0, rc - side // 2)
        rmax = min(h - 1, rmin + side - 1)
        cmin = max(0, cc - side // 2)
        cmax = min(w - 1, cmin + side - 1)

    return int(rmin), int(rmax), int(cmin), int(cmax)


def crop_and_resize(image: np.ndarray, box, size: int = IMAGE_SIZE[0]) -> np.ndarray:
    """Same crop+resize as convert_fibroid_data.py (PIL bilinear)."""
    rmin, rmax, cmin, cmax = box
    cropped = image[rmin:rmax + 1, cmin:cmax + 1]
    pil_img = PILimg.fromarray(cropped)
    resized = pil_img.resize((size, size), PILimg.BILINEAR)
    return np.array(resized)


_TO_TENSOR_NORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_image(rgb: np.ndarray):
    """Returns (tensor for the model, cropped+resized numpy image for display)."""
    box = compute_crop_box(rgb)
    cropped = crop_and_resize(rgb, box, size=IMAGE_SIZE[0])
    tensor = _TO_TENSOR_NORM(PILimg.fromarray(cropped)).unsqueeze(0)
    return tensor, cropped


# ============================================================================
# Model wrapper — same loading + inference pattern as push_node.py/diffusion_node.py
# ============================================================================

class DisplacementModel:
    def __init__(self, task: str, device: str):
        self.task = task
        self.device = device
        self.model: Optional[DisplacementDiffusionSDE] = None
        self._load()

    def _load(self):
        weights_path = Path(PACKAGE_PATH) / "models" / "diffusion_displacement" / "weights" / WEIGHT_FILES[self.task]
        if not weights_path.exists():
            raise FileNotFoundError(f"[{self.task}] weights not found: {weights_path}")

        ckpt = torch.load(weights_path, map_location=self.device)

        mc = ckpt["model_config"]
        cfg = DiffusionSDEConfig(**{
            k: v for k, v in mc.items() if k in DiffusionSDEConfig.__dataclass_fields__
        })
        model = DisplacementDiffusionSDE(cfg).to(self.device)

        sd = ckpt.get("ema_state_dict") or ckpt["model_state_dict"]
        model.load_state_dict(sd, strict=False)

        if "action_mean" in ckpt and "action_std" in ckpt:
            model.set_action_norm(ckpt["action_mean"].to(self.device), ckpt["action_std"].to(self.device))
        else:
            print(f"[{self.task}] WARNING: no action norm found in checkpoint")

        model.eval()
        self.model = model
        print(f"[{self.task}] model loaded ({weights_path.name})")

    @torch.no_grad()
    def predict_trajectories(self, image_tensor: torch.Tensor, start_point: np.ndarray,
                              n_samples: int = N_SAMPLES) -> np.ndarray:
        """Returns absolute waypoints [n_samples, N_WAYPOINTS, 3] (meters)."""
        img = image_tensor.expand(n_samples, -1, -1, -1).contiguous().to(self.device)
        sp = torch.from_numpy(start_point).float().unsqueeze(0).expand(n_samples, -1).contiguous().to(self.device)

        disps = self.model.predict(
            images=img,
            start_points=sp,
            num_steps=INFER_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            sampler=SAMPLER,
        )  # [n_samples, 30]

        disps = disps.cpu().numpy().reshape(n_samples, N_WAYPOINTS, 3)

        abs_traj = np.zeros_like(disps)
        abs_traj[:, 0] = start_point + disps[:, 0]
        for k in range(1, N_WAYPOINTS):
            abs_traj[:, k] = abs_traj[:, k - 1] + disps[:, k]

        return abs_traj


# ============================================================================
# Trajectory visualizer / selector (same pattern as push_node.py)
# ============================================================================

def visualize_and_select_trajectory(task: str, image: np.ndarray, trajectories: np.ndarray,
                                     start_point: np.ndarray) -> int:
    trajectories_mm = trajectories * 1000.0
    start_mm = start_point * 1000.0
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
    selected = {"idx": 0}

    def draw(cur):
        fig = plt.figure(figsize=(20, 10))
        ax3d = fig.add_subplot(121, projection="3d")
        ax3d.set_title(
            f"[{task}]  Trajectory {cur + 1}/{len(trajectories)}  —  Escape = cycle, Enter = confirm",
            fontsize=13,
        )

        for traj, col in zip(trajectories_mm, colors):
            ax3d.plot(*traj.T, alpha=0.2, linewidth=1.0, color=col)
            ax3d.scatter(*traj[-1], alpha=0.2, s=20, color=col)

        ht = trajectories_mm[cur]
        ax3d.plot(*ht.T, color="red", linewidth=3, label=f"Selected #{cur}")
        ax3d.scatter(*ht[-1], color="red", s=80, zorder=5)
        ax3d.scatter(*start_mm, color="lime", s=200, marker="o", label="START", zorder=5)

        fl = 15
        for vec, col, lbl in zip(np.eye(3) * fl, ["red", "green", "blue"], ["X", "Y", "Z"]):
            ax3d.quiver(*start_mm, *vec, color=col, linewidth=2, arrow_length_ratio=0.2)
            ax3d.text(*(start_mm + vec * 1.4), lbl, fontsize=9, color=col, fontweight="bold")

        ax3d.set_xlabel("X (mm)"); ax3d.set_ylabel("Y (mm)"); ax3d.set_zlabel("Z (mm)")
        ax3d.legend()

        ax_img = fig.add_subplot(122)
        if image is not None:
            ax_img.imshow(image)
        ax_img.set_title("Camera View"); ax_img.set_xticks([]); ax_img.set_yticks([])

        def on_key(event):
            if event.key == "enter":
                selected["idx"] = cur
                plt.close(fig)
            elif event.key == "escape":
                nxt = (cur + 1) % len(trajectories)
                selected["idx"] = nxt
                plt.close(fig)
                draw(nxt)

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.tight_layout()
        plt.show()

    draw(0)
    return selected["idx"]


# ============================================================================
# Node
# ============================================================================

class UterineDiffusionNode(Node):
    def __init__(self):
        super().__init__('uterine_diffusion_node')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self.device}")

        # Load whatever task models have weights available; skip the rest.
        self.task_models: Dict[str, DisplacementModel] = {}
        for task in TASK_NAMES:
            try:
                self.task_models[task] = DisplacementModel(task, self.device)
            except FileNotFoundError as e:
                self.get_logger().warn(f"Skipping task '{task}': {e}")

        if not self.task_models:
            raise RuntimeError(
                "No task models could be loaded — check weights under "
                "models/diffusion_displacement/weights/"
            )

        self.latest_image: Optional[np.ndarray] = None

        # Tracked tips from smoother_uterus
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.camera_sub = self.create_subscription(Image, CAMERA_TOPIC, self.camera_callback, 10)
        self.task_sub = self.create_subscription(String, CURRENT_TASK_TOPIC, self.task_callback, 10)

        self.action_pubs = {
            task: self.create_publisher(PointCloud2, ACTION_TOPICS[task], 10)
            for task in self.task_models
        }

        active = list(self.task_models.keys())
        self.get_logger().info(
            f"UterineDiffusionNode initialized. Active tasks: {active}. "
            f"Listening on {CURRENT_TASK_TOPIC} for dispatch."
        )

    # ------------- Callbacks -------------

    def camera_callback(self, msg: Image):
        img = np.frombuffer(msg.data, dtype=np.uint8)
        try:
            img = img.reshape((msg.height, msg.width, -1))
        except ValueError:
            self.get_logger().error(
                f"Image data size {img.size} does not match ({msg.height}, {msg.width}, 3)."
            )
            return
        self.latest_image = img

    def task_callback(self, msg: String):
        task = msg.data.strip()
        if task not in TASK_NAMES:
            # e.g. "screw", or any other non-diffusion task — not ours to handle.
            return
        self.run_inference_and_publish(task)

    def _get_start_point(self, task: str) -> Optional[np.ndarray]:
        """The arm's tracked tip from smoother_uterus, in that arm's model frame (FRAME_IDS)."""
        arm = ARM_FOR_TASK[task]
        try:
            t = self.tf_buffer.lookup_transform(FRAME_IDS[arm], TIP_FRAMES[arm], Time())
        except TransformException as e:
            self.get_logger().error(
                f"No tracked {arm} tip ({TIP_FRAMES[arm]} in {FRAME_IDS[arm]}): {e}. Is smoother_uterus running?")
            return None

        age = (self.get_clock().now() - Time.from_msg(t.header.stamp)).nanoseconds / 1e9
        if age > MAX_TIP_AGE_S:
            self.get_logger().error(f"Tracked {arm} tip is {age:.1f} s old; is smoother_uterus running?")
            return None

        p = t.transform.translation
        return np.array([p.x, p.y, p.z], dtype=np.float32)

    # ------------- Inference -------------

    def run_inference_and_publish(self, task: str):
        if task not in self.task_models:
            self.get_logger().error(f"Task '{task}' has no loaded model (missing weights?). Skipping.")
            return
        if self.latest_image is None:
            self.get_logger().error("Missing camera image; cannot run inference.")
            return

        start_point = self._get_start_point(task)
        arm = ARM_FOR_TASK[task]
        if start_point is None:
            self.get_logger().error(f"No start point for '{arm}' arm (see above); cannot run inference.")
            return

        self.get_logger().info(f"[{task}] Running uterine displacement-diffusion inference ({arm} arm)...")
        image = self.latest_image

        image_tensor, cropped_image = preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        trajectories = self.task_models[task].predict_trajectories(
            image_tensor, start_point, n_samples=N_SAMPLES
        )  # [N_SAMPLES, N_WAYPOINTS, 3]

        if INTERACTIVE_SELECT:
            sel = visualize_and_select_trajectory(task, cropped_image, trajectories, start_point)
        else:
            sel = 0

        selected_traj = trajectories[sel]  # [N_WAYPOINTS, 3]

        self.get_logger().info(
            f"[{task}] Selected trajectory #{sel} — start={selected_traj[0].round(4)} "
            f"end={selected_traj[-1].round(4)}"
        )

        self.action_pubs[task].publish(xyz_array_to_pointcloud2(selected_traj, frame_id=FRAME_IDS[arm]))
        self.get_logger().info(f"[{task}] Published {N_WAYPOINTS}-waypoint trajectory to {ACTION_TOPICS[task]}.")


def main(args=None):
    rclpy.init(args=args)
    node = UterineDiffusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()