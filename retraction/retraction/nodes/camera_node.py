#!/usr/bin/env python3
from __future__ import annotations

import threading
import time

import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class CameraNode(Node):
    """
    V4L2 capture via OpenCV instead of an ffmpeg subprocess.

    Why this over the ffmpeg version:
      - No dependency on how ffmpeg was compiled (this was the actual bug —
        your ffmpeg build lacks libv4l2 support, so -use_libv4l2 1 always
        failed to open the device).
      - No manual raw-byte framing over a pipe.
      - cv2.VideoCapture(..., cv2.CAP_V4L2) talks to the driver directly.
    """

    def __init__(self):
        super().__init__("camera_node")

        self.declare_parameter(
            "device",
            "/dev/v4l/by-id/usb-Hayear_Electronics_Co._Ltd_HAYEAR_CAMERA_SN00001-video-index0",
        )
        self.declare_parameter("width", 1280)
        self.declare_parameter("height", 720)
        self.declare_parameter("fps", 30)
        self.declare_parameter("fourcc", "MJPG")  # MJPG|YUYV
        self.declare_parameter("frame_id", "hy_optical")
        self.declare_parameter("restart_backoff_sec", 0.5)
        self.declare_parameter("qos_depth", 10)

        # Pixels to strip off each side to remove black border, applied
        # after capture at the *negotiated* resolution.
        self.declare_parameter("crop_top", 120)
        self.declare_parameter("crop_bottom", 200)
        self.declare_parameter("crop_left", 250)
        self.declare_parameter("crop_right", 319)

        self.device: str = self.get_parameter("device").value
        self.width: int = int(self.get_parameter("width").value)
        self.height: int = int(self.get_parameter("height").value)
        self.fps: int = int(self.get_parameter("fps").value)
        self.fourcc: str = str(self.get_parameter("fourcc").value)
        self.frame_id: str = self.get_parameter("frame_id").value
        self.base_backoff: float = float(self.get_parameter("restart_backoff_sec").value)
        self.backoff = self.base_backoff

        self.crop_top: int = int(self.get_parameter("crop_top").value)
        self.crop_bottom: int = int(self.get_parameter("crop_bottom").value)
        self.crop_left: int = int(self.get_parameter("crop_left").value)
        self.crop_right: int = int(self.get_parameter("crop_right").value)

        qos_depth = int(self.get_parameter("qos_depth").value)
        self.pub_img = self.create_publisher(Image, "image", qos_depth)
        self.bridge = CvBridge()

        self.cap: cv2.VideoCapture | None = None
        self._open_capture()

        self._stop_event = threading.Event()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        self.get_logger().info(
            f"CameraNode started. device={self.device} "
            f"{self.width}x{self.height}@{self.fps} fourcc={self.fourcc} "
            f"crop=(t{self.crop_top},b{self.crop_bottom},l{self.crop_left},r{self.crop_right})"
        )

    def _open_capture(self):
        if self.cap is not None:
            self.cap.release()

        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize latency

        if not cap.isOpened():
            self.get_logger().warn(f"Failed to open {self.device}, will retry.")
            self.cap = None
            return

        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        self.get_logger().info(
            f"Opened {self.device}: negotiated {actual_w:.0f}x{actual_h:.0f}@{actual_fps:.1f}"
        )
        self.cap = cap

    def _crop(self, frame):
        if not (self.crop_top or self.crop_bottom or self.crop_left or self.crop_right):
            return frame
        h, w = frame.shape[:2]
        y0 = self.crop_top
        y1 = h - self.crop_bottom
        x0 = self.crop_left
        x1 = w - self.crop_right
        if y1 <= y0 or x1 <= x0:
            self.get_logger().error(
                f"Crop margins exceed frame size ({w}x{h}); skipping crop this frame."
            )
            return frame
        return frame[y0:y1, x0:x1]

    def _capture_loop(self):
        # Runs on its own thread. No timer pacing this — cap.read() blocks
        # until the driver actually has a frame, so we publish the instant
        # each frame is ready instead of racing it against a separate clock.
        while not self._stop_event.is_set():
            if self.cap is None:
                time.sleep(self.backoff)
                self.backoff = min(5.0, self.backoff * 1.5)
                self._open_capture()
                continue

            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.get_logger().warn("Frame read failed, reopening device.")
                self.cap.release()
                self.cap = None
                continue

            self.backoff = self.base_backoff

            frame = self._crop(frame)

            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            self.pub_img.publish(msg)

    def destroy_node(self):
        self._stop_event.set()
        if self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()