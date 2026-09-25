import rclpy
from rclpy.node import Node
import threading

import torch
import numpy as np
from copy import deepcopy

from std_msgs.msg import Int8, Int32, Bool
from sensor_msgs.msg import PointCloud2, Image

import torch.nn as nn  # BatchNorm1d etc.
from ament_index_python.packages import get_package_share_directory

import torchvision.transforms as transforms
from PIL import Image as PILimg

import time
import pickle
import torch.nn.functional as F

from xgboost import XGBClassifier

# Slip detector utilities

import os
import open3d as o3d
import argparse
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.patches import ConnectionPatch

PACKAGE_PATH = get_package_share_directory('retraction')
CUR_PATH = os.path.dirname(__file__)

import sys
sys.path.append(os.path.join(CUR_PATH, '..'))

from point_cloud_utils.point_cloud_utils import *
# from point_cloud_utils.utils import *

# Import the Push2dCVAE model using relative import
from ..models.RetractModels.push2dCVAE import Push2dCVAE
from ..models.RetractModels.UnsupervisedObserver import ImageBackboneEncoder, apply_circular_roi



def crop_image(image, xmin=650, xmax=1250, ymin=250, ymax=850):
    h, w = image.shape[:2]
    x0 = max(0, xmin); x1 = min(w, xmax)
    y0 = max(0, ymin); y1 = min(h, ymax)
    if x0 >= x1 or y0 >= y1:
        print(f"[crop_image] Invalid crop box ({xmin},{xmax},{ymin},{ymax}) for image size {w}x{h}, returning original.")
        return image
    return image[y0:y1, x0:x1]


def preprocess_image(image, input_size=(224, 224)):
    """Preprocess the image for the model."""
    # crop the image first
    image = crop_image(image)

    pil_image = PILimg.fromarray(image)
    # Define transformations (adjust based on how your model was trained)
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        img = image

        print(img)
        if img is not None:
            plt.figure("retract_inference/latest_image")
            if img.ndim == 3 and img.shape[2] == 3:
                plt.imshow(img)  # assumes RGB
            else:
                plt.imshow(img, cmap="gray")
            plt.axis("off")
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print("FAIL")
    
    return transform(pil_image).unsqueeze(0)  # Add batch dimension



# ===========================
# BoostSlip inference helper
# ===========================
class BoostSlipDetector:
    """
    Lightweight inference wrapper for the XGBoost slip classifier trained in BoostSlip.py.

    Feature vector matches training:
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

        # Backbone encoder (same as training)
        self.encoder = ImageBackboneEncoder(backbone=self.backbone, train_backbone=False).to(self.device)
        self.encoder.eval()

        # XGBoost model
        self.clf = XGBClassifier()
        self.clf.load_model(self.model_path)

        # Feature normalization stats (optional but recommended)
        self.mu = None
        self.sigma = None
        if os.path.exists(self.stats_path):
            stats = np.load(self.stats_path)
            # Be robust to key names
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

        # Image preprocessing must match training (224x224 normalized ImageNet)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _img_to_tensor(self, img_bgr_uint8: np.ndarray) -> torch.Tensor:
        # Node stores OpenCV images (BGR). Convert to RGB for PIL/torchvision.
        img_rgb = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2RGB)
        pil = PILimg.fromarray(img_rgb)
        t = self.transform(pil).unsqueeze(0).to(self.device)  # [1,C,H,W]
        return t

    @torch.no_grad()
    def predict(self, img_pre_bgr: np.ndarray, img_post_bgr: np.ndarray):
        """
        Returns: (is_slip: bool, slip_prob: float)
        """
        x_pre = self._img_to_tensor(img_pre_bgr)
        x_post = self._img_to_tensor(img_post_bgr)

        # Circular ROI (same as training)
        x_pre = apply_circular_roi(x_pre, self.roi_center, self.roi_radius)
        x_post = apply_circular_roi(x_post, self.roi_center, self.roi_radius)

        z_pre = self.encoder(x_pre)   # [1, D]
        z_post = self.encoder(x_post) # [1, D]

        diff = z_pre - z_post
        diff_abs = torch.abs(diff)
        diff_sq = diff ** 2

        cos_sim = F.cosine_similarity(z_pre, z_post, dim=-1, eps=1e-8)  # [1]
        cos_dist = (1.0 - cos_sim).unsqueeze(-1)  # [1,1]
        l2_dist = torch.norm(diff, dim=-1, keepdim=True)  # [1,1]

        feat = torch.cat([z_pre, z_post, diff_abs, diff_sq, cos_dist, l2_dist], dim=-1).cpu().numpy()  # [1, Dtot]

        if self.mu is not None and self.sigma is not None:
            denom = np.where(self.sigma == 0, 1.0, self.sigma)
            feat = (feat - self.mu) / denom

        # predict_proba returns [p(class0), p(class1)] in the order of classes seen during training
        proba = self.clf.predict_proba(feat)[0]
        slip_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
        is_slip = slip_prob >= self.threshold
        return is_slip, slip_prob


class RetractFib(Node):
    def __init__(self):
        super().__init__('RetractFib')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        print("Current Path", CUR_PATH)
        print("Package Path", PACKAGE_PATH)

        # ------------------------
        #   MODEL SETUP
        # ------------------------
        self.cvae = Push2dCVAE(
            n_latent_dims=7,        # Latent space dimension
            image_feature_dim=64,   # Dimension of image features after encoding
            backbone='eff',    # Using ResNet50 as backbone
            device=self.device
        ).to(self.device)

        weights_path = os.path.join(
            PACKAGE_PATH,
            'models', 'RetractModels', 'weights',
            'fib_retract_best_eff.pth'
        )

        # Load weights
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found at {weights_path}")

        self.cvae.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.cvae.eval()  # Set to evaluation mode

        # ------------------------
        #   STATE / SUBSCRIBERS
        # ------------------------
        self.latest_image = None
        self.converged = False  # if you ever want to track convergence

        # Camera subscription: 2D image channel
        self.camera_sub = self.create_subscription(
            Image,
            '/hy_camera/image',
            self.camera_callback,
            10
        )

        # ------------------------
        #   PUBLISHERS
        # ------------------------
        # Single combined action topic: start + end points
        self.retract_action_pub = self.create_publisher(
            PointCloud2,
            '/retract_action',
            10
        )

        # (Optional) Keep these if you still like visualizing them separately
        self.start_pt_pub = self.create_publisher(
            PointCloud2,
            '/start_pt',
            10
        )

        self.end_pt_pub = self.create_publisher(
            PointCloud2,
            '/end_pt',
            10
        )

        # ------------------------
        #   BOOST SLIP SETTINGS
        # ------------------------
        self.boost_enabled = True
        self.boost_max_attempts = 5
        self.start_capture_delay_s = 0.8   # time after publishing action to snapshot 'start' image
        self.end_capture_delay_s = 0.2     # time after /done_retract to snapshot 'end' image
        self.action_timeout_s = 20.0       # wait for /done_retract
        self.slip_threshold = 0.5

        # Will be set each attempt by run_inference_and_publish()
        self.last_action_start = None
        self.last_action_end = None

        # Done signal for motion completion (published by your executor stack)
        self._done_evt = threading.Event()
        self.done_sub = self.create_subscription(
            Bool,
            '/done_retract',
            self.done_callback,
            10
        )

        # Slip detector init (XGBoost + ROI encoder)
        try:
            stats_path = os.path.join(
            PACKAGE_PATH,
            'models', 'RetractModels', 'weights',
            'xgb_features_stats.npz'
        )
            model_path = os.path.join(
            PACKAGE_PATH,
            'models', 'RetractModels', 'weights',
            'xgb_slip_classifier.json '
        )
            self.slip_detector = BoostSlipDetector(
                model_path=model_path,
                stats_path=stats_path,
                backbone='eff',
                roi_center=(112, 112),
                roi_radius=112,
                device=self.device,
                threshold=self.slip_threshold,
            )
            self.get_logger().info(f"BoostSlip detector loaded: {model_path}")
        except Exception as e:
            self.slip_detector = None
            self.boost_enabled = False
            self.get_logger().error(f"BoostSlip disabled (failed to load model/stats): {e}")

        self.get_logger().info(
            "RetractFib node initialized. Press ENTER in the terminal to run a retraction inference."
        )

    # ----------------------------------------------------
    #   CALLBACKS
    # ----------------------------------------------------
    def camera_callback(self, msg: Image):
        """
        Store the latest camera image as an HxWx3 numpy array.
        Adjust reshape/encoding if your camera is not 1080x1080 BGR.
        """
        # Convert from raw bytes to numpy
        img = np.frombuffer(msg.data, dtype=np.uint8)

        try:
            img = img.reshape((msg.height, msg.width, -1))
        except ValueError:
            self.get_logger().error(
                f"Image data size {img.size} does not match expected shape "
                f"({msg.height}, {msg.width}, 3)."
            )
            return

        # If encoding is BGR and you want RGB for torchvision:
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.latest_image = img
        # self.get_logger().info('Received camera image')  # can be noisy

    # ----------------------------------------------------
    #   CORE INFERENCE LOGIC
    # ----------------------------------------------------
    def done_callback(self, msg: Bool):
        # Expect True when the motion executor has finished the retract action
        if bool(msg.data):
            self._done_evt.set()

    def get_push_action(self, image):
        """
        Get the push action from the 2D image.

        NEW BEHAVIOR:
        - The model now outputs a displacement.
        - Assume each sampled action is [start, displacement] (each 3D).
        - We convert displacement -> endpoint = start + displacement.
        - We pick the "best" action based on displacement magnitude with
          some direction constraints (like before).
        """
        device = self.device

        input_tensor = preprocess_image(image)
        input_tensor = input_tensor.to(device)

        # Forward pass: sample multiple actions
        with torch.no_grad():
            # Shape assumed: (N, 2, 3) where [0] = start, [1] = displacement
            sampled_actions = self.cvae.sample(input_tensor, n_samples=100, device=device)

        # Move to CPU for numpy ops
        sampled_actions = np.array(sampled_actions)

        best_idx = None
        best_norm = -float('inf')
        best_start = None
        best_end = None

        self.get_logger().warn(
            "No actions satisfied direction constraints; using first sampled action."
        )
        start = sampled_actions[0, 0]
        disp = sampled_actions[0, 1]
        end = start + disp

        self.get_logger().info(
            f"Selected action with displacement norm {np.linalg.norm(end - start):.4f} m"
        )
        self.get_logger().info(f"Start: {start}, End: {end}")

        return start, end

    def run_inference_and_publish(self, block_vis: bool = True):
        """
        Entry point for the 'button click' (ENTER key).
        - Uses latest_image
        - Runs model inference
        - Publishes /retract_action as PointCloud2 containing [start; end]
        """
        if self.latest_image is None:
            self.get_logger().error('Missing camera image; cannot compute action.')
            return
        
        self.get_logger().info('Running retraction inference...')
        start_point, end_point = self.get_push_action(self.latest_image)
        # Store for BoostSlip wrapper
        self.last_action_start = np.array(start_point, copy=True)
        self.last_action_end = np.array(end_point, copy=True)
        if hasattr(self, '_done_evt'):
            self._done_evt.clear()


        start_point_array = start_point.reshape(-1, 3)
        end_point_array = end_point.reshape(-1, 3)

        # Optional separate debug topics
        start_pc2 = xyz_array_to_pointcloud2(start_point_array)
        end_pc2 = xyz_array_to_pointcloud2(end_point_array)
        self.start_pt_pub.publish(start_pc2)
        self.end_pt_pub.publish(end_pc2)

        # Concatenate start and end into a single action list
        actions_list = np.vstack([start_point_array, end_point_array])  # shape (2, 3)

        action_list_msg = xyz_array_to_pointcloud2(actions_list)
        self.retract_action_pub.publish(action_list_msg)

        self.get_logger().info("Published /retract_action PointCloud2 (start + end).")

    # ----------------------------------------------------
    #   OPTIONAL: IMAGE + 3D VISUALIZATION (UNCHANGED)
    # ----------------------------------------------------
    def xyz_to_pixel(self, point: np.ndarray) -> list:
        K = np.array(self._camera_info.k).reshape(3, 3)
        fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
        x, y, z = point
        px = x * fx / z + cx
        py = y * fy / z + cy
        return [int(px), int(py)]

    def visualize_action_on_image(self, action, image):
        """
        Visualize push action in 3D robot frame with camera image on the side.
        (You can still call this manually if you want, but it's no longer used to
        select the action.)
        """
        action_start = action[0]
        action_end = action[1]
        print(f"Robot frame action_start: {action_start}")
        print(f"Robot frame action_end: {action_end}")

        fig = plt.figure(figsize=(20, 10))

        # Left subplot: 3D visualization in robot frame
        ax_3d = fig.add_subplot(121, projection='3d')
        ax_3d.set_title("3D Robot Frame - Push Action", fontsize=14, pad=20)

        ax_3d.scatter(*action_start, color='green', s=200, alpha=0.8, label='START')
        ax_3d.scatter(*action_end, color='red', s=200, alpha=0.8, label='END')

        ax_3d.plot(
            [action_start[0], action_end[0]],
            [action_start[1], action_end[1]],
            [action_start[2], action_end[2]],
            color='blue', linewidth=4, alpha=0.9
        )

        frame_length = 0.015
        origin = [0, 0, 0]

        ax_3d.quiver(
            origin[0], origin[1], origin[2],
            frame_length, 0, 0,
            color='red', linewidth=2, arrow_length_ratio=0.2, label='X-axis'
        )
        ax_3d.quiver(
            origin[0], origin[1], origin[2],
            0, frame_length, 0,
            color='green', linewidth=2, arrow_length_ratio=0.2, label='Y-axis'
        )
        ax_3d.quiver(
            origin[0], origin[1], origin[2],
            0, 0, frame_length,
            color='blue', linewidth=2, arrow_length_ratio=0.2, label='Z-axis'
        )

        ax_3d.text(frame_length + 0.005, 0, 0, 'X', fontsize=10, color='red', fontweight='bold')
        ax_3d.text(0, frame_length + 0.005, 0, 'Y', fontsize=10, color='green', fontweight='bold')
        ax_3d.text(0, 0, frame_length + 0.005, 'Z', fontsize=10, color='blue', fontweight='bold')

        ax_3d.text(action_start[0], action_start[1], action_start[2] + 0.01,
                   'START', fontsize=12, color='green')
        ax_3d.text(action_end[0], action_end[1], action_end[2] + 0.01,
                   'END', fontsize=12, color='red')

        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        ax_3d.legend()

        max_range = 0.1
        center_x = (action_start[0] + 0) / 2
        center_y = (action_start[1] + 0) / 2
        center_z = (action_start[2] + 0) / 2

        ax_3d.set_xlim([center_x - max_range, center_x + max_range])
        ax_3d.set_ylim([center_y - max_range, center_y + max_range])
        ax_3d.set_zlim([center_z - max_range, center_z + max_range])

        ax_img = fig.add_subplot(122)
        ax_img.imshow(image)
        ax_img.set_title("Camera View", fontsize=14, pad=20)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        coord_text = f"Robot Frame Coordinates:\n"
        coord_text += f"Start: ({action_start[0]:.4f}, {action_start[1]:.4f}, {action_start[2]:.4f})\n"
        coord_text += f"End: ({action_end[0]:.4f}, {action_end[1]:.4f}, {action_end[2]:.4f})\n"
        coord_text += f"Distance: {np.linalg.norm(action_end - action_start):.4f}m"

        ax_img.text(
            0.02, 0.98, coord_text, transform=ax_img.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8)
        )

        plt.tight_layout()

        plt.show()

    def run_inference_and_publish_boost(self):
        """
        Same as run_inference_and_publish(), but:
          - publish action
          - snapshot start image (after a short delay)
          - wait for /done_retract
          - snapshot end image
          - run BoostSlip detector
          - retry if slip
        """
        if self.latest_image is None:
            self.get_logger().error('Missing camera image; cannot compute action.')
            return

        if (not getattr(self, "boost_enabled", False)) or (self.slip_detector is None):
            # Fallback to original behavior
            self.run_inference_and_publish(block_vis=True)
            return

        for attempt in range(1, int(self.boost_max_attempts) + 1):
            self.get_logger().info(f"[BoostSlip] Attempt {attempt}/{self.boost_max_attempts}")

            # Publish + visualize (non-blocking so we can keep executing)
            self.run_inference_and_publish(block_vis=False)

            # Snapshot "start" image (after giving the system a moment to move/settle at start)
            time.sleep(float(self.start_capture_delay_s))
            if self.latest_image is None:
                self.get_logger().error("[BoostSlip] Missing camera image after publish; aborting.")
                return
            start_img = np.array(self.latest_image, copy=True)

            # Wait for motion completion
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
                # Show the plot window for the user now that we're done
                plt.show()
                return

            self.get_logger().warn("[BoostSlip] Slip detected — retrying with a new action...")

        self.get_logger().warn("[BoostSlip] Reached max attempts; keeping the last action (even if slip).")
        plt.show()

def keyboard_thread(node: RetractFib):
    """
    Separate thread that waits for ENTER in the terminal.
    Each ENTER press triggers a retraction inference + publish.
    """
    while rclpy.ok():
        try:
            input(">> Press ENTER to run retraction inference (Ctrl+C to quit)...\n")
        except EOFError:
            # Terminal closed or stdin gone
            break
        if not rclpy.ok():
            break
        node.run_inference_and_publish_boost()


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Push2d node: press ENTER to run retraction inference and publish /retract_action.'
    )
    # If you have custom args, parse them here
    _ = parser.parse_args()

    rclpy.init(args=args)
    node = RetractFib()

    # Start keyboard listener in a separate thread
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