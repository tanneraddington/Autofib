#!/usr/bin/env python3
"""
RetractFibDiffusion node:
- Same behavior as RetractFib (BoostSlip + topics + publishing),
- but uses RetractDiffV diffusion model instead of Push2dCVAE.
"""

import rclpy
from rclpy.node import Node
import threading

import torch
import numpy as np

from std_msgs.msg import Bool
from sensor_msgs.msg import PointCloud2, Image

from ament_index_python.packages import get_package_share_directory

import torchvision.transforms as transforms
from PIL import Image as PILimg

import time
import json
import torch.nn.functional as F

from xgboost import XGBClassifier

import os
import argparse
import cv2
import matplotlib.pyplot as plt

import sys

PACKAGE_PATH = get_package_share_directory('retraction')
CUR_PATH = os.path.dirname(__file__)

sys.path.append(os.path.join(CUR_PATH, '..'))

from point_cloud_utils.point_cloud_utils import *  # xyz_array_to_pointcloud2, etc.

# ---- Diffusion model import (this MUST match where you placed the file in your package) ----
from ..models.RetractModels.retract_diffusion_vpred import DiffusionSchedule, RetractDiffV

# Slip detector utilities (unchanged)
from ..models.RetractModels.UnsupervisedObserver import ImageBackboneEncoder, apply_circular_roi


# xmin=650, xmax=1250, ymin=250, ymax=850
def crop_image(image, xmin=650, xmax=1250, ymin=250, ymax=850):
    h, w = image.shape[:2]
    x0 = max(0, xmin); x1 = min(w, xmax)
    y0 = max(0, ymin); y1 = min(h, ymax)
    if x0 >= x1 or y0 >= y1:
        print(f"[crop_image] Invalid crop box ({xmin},{xmax},{ymin},{ymax}) for image size {w}x{h}, returning original.")
        return image
    return image[y0:y1, x0:x1]


def preprocess_image(image, input_size=(224, 224)):
    """
    Preprocess image -> torch tensor [1,3,224,224] normalized.
    NOTE: This includes crop_image(). If diffusion training did NOT crop,
          remove the crop step for best match.
    """
    image = crop_image(image)

    pil_image = PILimg.fromarray(image)
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Optional debug viz (WARNING: plt.show() is blocking)
    # If you want non-blocking, use plt.pause(0.001) instead.
    try:
        img = image
        if img is not None:
            plt.figure("retract_inference/latest_image")
            if img.ndim == 3 and img.shape[2] == 3:
                plt.imshow(img)  # assumes RGB
            else:
                plt.imshow(img, cmap="gray")
            plt.axis("off")
            plt.tight_layout()
            plt.show()
    except Exception:
        pass

    return transform(pil_image).unsqueeze(0)


# ===========================
# Action normalization loader
# ===========================
class ActionNorm:
    """
    Used by diffusion trainer:
    action = [start(3), disp(3)] in meters
    normalize: (x - mean) / std
    unnormalize: x * std + mean
    """
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-8):
        self.mean = mean.float()
        self.std = std.float().clamp_min(eps)

    @staticmethod
    def load(path: str, device: torch.device):
        with open(path, "r") as f:
            d = json.load(f)
        mean = torch.tensor(d["mean"], dtype=torch.float32, device=device)
        std = torch.tensor(d["std"], dtype=torch.float32, device=device)
        return ActionNorm(mean, std)

    def unnormalize(self, x_n: torch.Tensor) -> torch.Tensor:
        return x_n * self.std.to(x_n.device) + self.mean.to(x_n.device)


# ===========================
# BoostSlip inference helper
# ===========================
class BoostSlipDetector:
    """
    Feature vector:
      [z_pre, z_post, |diff|, diff^2, cosine_distance, l2_distance]
    """
    def __init__(
        self,
        model_path: str,
        stats_path: str,
        backbone: str = "eff",
        roi_center=(112, 112),
        roi_radius=112,
        device: str = "cpu",
        threshold: float = 0.5,
    ):
        self.model_path = model_path
        self.stats_path = stats_path
        self.backbone = backbone
        self.roi_center = tuple(roi_center)
        self.roi_radius = int(roi_radius)
        self.device = device
        self.threshold = float(threshold)

        self.encoder = ImageBackboneEncoder(backbone=self.backbone, train_backbone=False).to(self.device)
        self.encoder.eval()

        self.clf = XGBClassifier()
        self.clf.load_model(self.model_path)

        self.mu = None
        self.sigma = None
        if os.path.exists(self.stats_path):
            stats = np.load(self.stats_path)
            for mk in ["mean", "mu", "x_mean"]:
                if mk in stats:
                    self.mu = stats[mk]
                    break
            for sk in ["std", "sigma", "x_std"]:
                if sk in stats:
                    self.sigma = stats[sk]
                    break
            if self.mu is None and "arr_0" in stats:
                self.mu = stats["arr_0"]
            if self.sigma is None and "arr_1" in stats:
                self.sigma = stats["arr_1"]

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _img_to_tensor(self, img_bgr_uint8: np.ndarray) -> torch.Tensor:
        img_rgb = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2RGB)
        pil = PILimg.fromarray(img_rgb)
        t = self.transform(pil).unsqueeze(0).to(self.device)
        return t

    @torch.no_grad()
    def predict(self, img_pre_bgr: np.ndarray, img_post_bgr: np.ndarray):
        x_pre = self._img_to_tensor(img_pre_bgr)
        x_post = self._img_to_tensor(img_post_bgr)

        x_pre = apply_circular_roi(x_pre, self.roi_center, self.roi_radius)
        x_post = apply_circular_roi(x_post, self.roi_center, self.roi_radius)

        z_pre = self.encoder(x_pre)
        z_post = self.encoder(x_post)

        diff = z_pre - z_post
        diff_abs = torch.abs(diff)
        diff_sq = diff ** 2

        cos_sim = F.cosine_similarity(z_pre, z_post, dim=-1, eps=1e-8)
        cos_dist = (1.0 - cos_sim).unsqueeze(-1)
        l2_dist = torch.norm(diff, dim=-1, keepdim=True)

        feat = torch.cat([z_pre, z_post, diff_abs, diff_sq, cos_dist, l2_dist], dim=-1).cpu().numpy()

        if self.mu is not None and self.sigma is not None:
            denom = np.where(self.sigma == 0, 1.0, self.sigma)
            feat = (feat - self.mu) / denom

        proba = self.clf.predict_proba(feat)[0]
        slip_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
        is_slip = slip_prob >= self.threshold
        return is_slip, slip_prob


class RetractFibDiffusion(Node):
    def __init__(self):
        super().__init__('RetractFibDiffusion')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_str = "cuda" if self.device.type == "cuda" else "cpu"
        self.get_logger().info(f"Using device: {self.device_str}")

        # ------------------------
        #   DIFFUSION MODEL SETUP
        # ------------------------
        schedule = DiffusionSchedule(T=1000, schedule="cosine")
        self.diffuser = RetractDiffV(
            backbone="efficientnet_b0",
            action_dim=6,
            cond_dim=256,
            schedule=schedule,
            denoiser_hidden=512,
            denoiser_depth=4,
            denoiser_dropout=0.05,
            cfg_drop_prob=0.1,
            lambda_x0=0.5,
            start_weight=2.0,
            end_weight=1.0,
            use_snr_weight=True,
        ).to(self.device)
        self.diffuser.eval()

        # ---- weights + norm paths (must exist) ----
        weights_dir = os.path.join(PACKAGE_PATH, 'models', 'RetractModels', 'weights')
        ckpt_path = os.path.join(weights_dir, 'retract_diffuser_vpred_best.pth')
        norm_path = os.path.join(weights_dir, 'action_norm.json')

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Diffusion checkpoint not found at {ckpt_path}")
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"Action norm not found at {norm_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)

        # Load EMA if present
        if isinstance(ckpt, dict) and "ema" in ckpt:
            self.diffuser.load_state_dict(ckpt["ema"], strict=True)
            self.get_logger().info("Loaded diffusion EMA weights.")
        elif isinstance(ckpt, dict) and "model" in ckpt:
            self.diffuser.load_state_dict(ckpt["model"], strict=True)
            self.get_logger().warn("Loaded diffusion TRAIN weights (no EMA found).")
        else:
            self.diffuser.load_state_dict(ckpt, strict=True)
            self.get_logger().warn("Loaded diffusion state_dict (no wrapper keys).")

        self.action_norm = ActionNorm.load(norm_path, device=self.device)
        self.get_logger().info(f"Loaded action_norm.json from {norm_path}")

        # Sampling params
        self.sampler = "ddim"
        self.n_steps = 250
        self.guidance_scale = 1.5
        self.eta = 0.0
        self.k_candidates = 1

        # ------------------------
        #   STATE / SUBSCRIBERS
        # ------------------------
        self.latest_image = None

        self.camera_sub = self.create_subscription(
            Image,
            '/hy_camera/image',
            self.camera_callback,
            10
        )

        # ------------------------
        #   PUBLISHERS
        # ------------------------
        self.retract_action_pub = self.create_publisher(PointCloud2, '/retract_action', 10)
        self.start_pt_pub = self.create_publisher(PointCloud2, '/start_pt', 10)
        self.end_pt_pub = self.create_publisher(PointCloud2, '/end_pt', 10)

        # ------------------------
        #   BOOST SLIP SETTINGS
        # ------------------------
        self.boost_enabled = True
        self.boost_max_attempts = 5
        self.start_capture_delay_s = 0.8
        self.end_capture_delay_s = 0.2
        self.action_timeout_s = 20.0
        self.slip_threshold = 0.5

        self._done_evt = threading.Event()
        self.done_sub = self.create_subscription(Bool, '/done_retract', self.done_callback, 10)

        # Slip detector init
        try:
            stats_path = os.path.join(PACKAGE_PATH, 'models', 'RetractModels', 'weights', 'xgb_features_stats.npz')
            model_path = os.path.join(PACKAGE_PATH, 'models', 'RetractModels', 'weights', 'xgb_slip_classifier.json')  # no trailing space

            self.slip_detector = BoostSlipDetector(
                model_path=model_path,
                stats_path=stats_path,
                backbone='eff',
                roi_center=(112, 112),
                roi_radius=112,
                device=self.device_str,   # IMPORTANT: "cuda" / "cpu"
                threshold=self.slip_threshold,
            )
            self.get_logger().info(f"BoostSlip detector loaded: {model_path}")
        except Exception as e:
            self.slip_detector = None
            self.boost_enabled = False
            self.get_logger().error(f"BoostSlip disabled (failed to load model/stats): {e}")

        self.get_logger().info("RetractFibDiffusion initialized. Press ENTER to run inference.")

    # ----------------------------------------------------
    #   CALLBACKS
    # ----------------------------------------------------
    def camera_callback(self, msg: Image):
        img = np.frombuffer(msg.data, dtype=np.uint8)
        try:
            img = img.reshape((msg.height, msg.width, -1))
        except ValueError:
            self.get_logger().error(
                f"Image data size {img.size} does not match expected shape "
                f"({msg.height}, {msg.width}, 3)."
            )
            return
        self.latest_image = img

    def done_callback(self, msg: Bool):
        if bool(msg.data):
            self._done_evt.set()

    # ----------------------------------------------------
    #   DIFFUSION ACTION LOGIC
    # ----------------------------------------------------
    @torch.no_grad()
    def get_push_action(self, image_bgr: np.ndarray):
        if image_bgr is None:
            raise RuntimeError("No image provided to get_push_action().")

        x = preprocess_image(image_bgr).to(self.device)  # [1,3,224,224]

        candidates_real = []
        for _ in range(int(self.k_candidates)):
            x0_n = self.diffuser.sample_actions(
                x,
                n_steps=int(self.n_steps),
                guidance_scale=float(self.guidance_scale),
                sampler=str(self.sampler),
                eta=float(self.eta),
            )  # (1,6) normalized

            x0_real = self.action_norm.unnormalize(x0_n)  # (1,6) meters
            candidates_real.append(x0_real)

        acts = torch.cat(candidates_real, dim=0)   # (K,6)
        a = acts[0].detach().cpu().numpy()

        start = a[:3].copy()
        disp = a[3:].copy()
        end = start + disp

        self.get_logger().info(f"[Diffusion] start={start} disp={disp} ||disp||={float(np.linalg.norm(disp)):.4f} m")
        self.get_logger().info(f"[Diffusion] end={end}")

        return start, end

    def run_inference_and_publish(self, block_vis: bool = True):
        if self.latest_image is None:
            self.get_logger().error('Missing camera image; cannot compute action.')
            return

        self.get_logger().info('Running retraction inference (diffusion)...')
        start_point, end_point = self.get_push_action(self.latest_image)

        self._done_evt.clear()

        start_point_array = start_point.reshape(-1, 3)
        end_point_array = end_point.reshape(-1, 3)

        self.start_pt_pub.publish(xyz_array_to_pointcloud2(start_point_array))
        self.end_pt_pub.publish(xyz_array_to_pointcloud2(end_point_array))

        actions_list = np.vstack([start_point_array, end_point_array])
        self.retract_action_pub.publish(xyz_array_to_pointcloud2(actions_list))

        self.get_logger().info("Published /retract_action PointCloud2 (start + end).")

    def run_inference_and_publish_boost(self):
        if self.latest_image is None:
            self.get_logger().error('Missing camera image; cannot compute action.')
            return

        if (not getattr(self, "boost_enabled", False)) or (self.slip_detector is None):
            self.run_inference_and_publish(block_vis=True)
            return

        for attempt in range(1, int(self.boost_max_attempts) + 1):
            self.get_logger().info(f"[BoostSlip] Attempt {attempt}/{self.boost_max_attempts}")

            self.run_inference_and_publish(block_vis=False)

            time.sleep(float(self.start_capture_delay_s))
            if self.latest_image is None:
                self.get_logger().error("[BoostSlip] Missing camera image after publish; aborting.")
                return
            start_img = np.array(self.latest_image, copy=True)

            done = self._done_evt.wait(timeout=float(self.action_timeout_s))
            if not done:
                self.get_logger().warn("[BoostSlip] Timed out waiting for /done_retract; using latest image as end image.")

            time.sleep(float(self.end_capture_delay_s))
            if self.latest_image is None:
                self.get_logger().error("[BoostSlip] Missing camera image at end; aborting.")
                return
            end_img = np.array(self.latest_image, copy=True)

            try:
                is_slip, slip_prob = self.slip_detector.predict(start_img, end_img)
            except Exception as e:
                self.get_logger().error(f"[BoostSlip] Slip detector failed: {e}")
                return

            self.get_logger().info(f"[BoostSlip] slip_prob={slip_prob:.3f} (thr={self.slip_threshold:.2f})")

            if not is_slip:
                self.get_logger().info("[BoostSlip] No slip detected. Keeping this action.")
                plt.show()
                return

            self.get_logger().warn("[BoostSlip] Slip detected — retrying with a new action...")

        self.get_logger().warn("[BoostSlip] Reached max attempts; keeping the last action (even if slip).")
        plt.show()


def keyboard_thread(node):
    while rclpy.ok():
        try:
            input(">> Press ENTER to run retraction inference (Ctrl+C to quit)...\n")
        except EOFError:
            break
        if not rclpy.ok():
            break
        node.run_inference_and_publish_boost()


def main(args=None):
    parser = argparse.ArgumentParser(
        description='RetractFibDiffusion: press ENTER to run diffusion inference and publish /retract_action.'
    )
    _ = parser.parse_args()

    rclpy.init(args=args)
    node = RetractFibDiffusion()   # IMPORTANT: use diffusion node

    kb_thread = threading.Thread(target=keyboard_thread, args=(node,), daemon=True)
    kb_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
