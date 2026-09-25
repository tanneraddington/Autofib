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


def crop_image(image, xmin=650, xmax=1250, ymin=250, ymax=850):
    """
    Crop the image to the given bounding box.
    Args:
        image: HxWxC numpy array (BGR, as from OpenCV).
        xmin, xmax, ymin, ymax: pixel coordinates in (x, y) in the ORIGINAL image.
    Returns:
        Cropped image as numpy array.
    """
    h, w = image.shape[:2]

    x0 = max(0, xmin)
    x1 = min(w, xmax)
    y0 = max(0, ymin)
    y1 = min(h, ymax)

    if x0 >= x1 or y0 >= y1:
        print(f"[crop_image] Invalid crop box ({xmin},{xmax},{ymin},{ymax}) for image size {w}x{h}, returning original.")
        return image

    # OpenCV: [rows, cols] = [y, x]
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
    return transform(pil_image).unsqueeze(0)  # Add batch dimension


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
            backbone='resnet50',    # Using ResNet50 as backbone
            device=self.device
        ).to(self.device)

        weights_path = os.path.join(
            PACKAGE_PATH,
            'models', 'RetractModels', 'weights',
            'fib_retract_best_epoch12.pth'
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

        for i in range(sampled_actions.shape[0]):
            start = sampled_actions[i, 0]        # (3,)
            disp = sampled_actions[i, 1]        # (3,)
            end = start + disp                           # displacement -> endpoint

            norm = np.linalg.norm(disp)
            if norm > best_norm:
                best_norm = norm
                best_idx = i
                best_start = start
                best_end = end

        if best_idx is None:
            # fallback: just take the first sample, still treating index 1 as displacement
            self.get_logger().warn(
                "No actions satisfied direction constraints; using first sampled action."
            )
            start = sampled_actions[0, 0].numpy()
            disp = sampled_actions[0, 1].numpy()
            end = start + disp
        else:
            start = best_start
            end = best_end

        self.get_logger().info(
            f"Selected action with displacement norm {np.linalg.norm(end - start):.4f} m"
        )
        self.get_logger().info(f"Start: {start}, End: {end}")

        return start, end

    def run_inference_and_publish(self):
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
        node.run_inference_and_publish()


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
