#!/usr/bin/env python3
import os
import pickle
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge

# =======================
# ---- USER MACROS ------
# =======================
IMAGE_TOPIC      = "/hy_camera/image"
LEFT_POSE_TOPIC  = "/fwkin_l"
RIGHT_POSE_TOPIC = "/fwkin_r"
OUTPUT_DIR       = "data/fibroid_live/parsed_pickles_cc"
WINDOW_NAME      = "Data Collector"
DISPLAY_FPS      = 20

# When recording between S and E:
RECORD_FPS = 10          # target save rate (<= camera FPS); set None to save every received frame
SAVE_IMAGES = True       # set False if you only want pose + timestamp (but usually True)
# =======================


def save_to_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved to {filename}")


def pose_to_pos_array(msg: Pose):
    """Return a 3D numpy array [x, y, z] or None (matches bag pickle format)."""
    if msg is None:
        return None
    return np.array(
        [float(msg.position.x), float(msg.position.y), float(msg.position.z)],
        dtype=np.float32,
    )


class DataCollector(Node):
    def __init__(self):
        super().__init__("data_collector")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.bridge = CvBridge()

        # Buffers (latest)
        self.latest_rgb = None
        self.latest_stamp = 0.0
        self.latest_left_pose = None
        self.latest_right_pose = None

        # Recording state
        self.recording = False
        self.record_id = 100         # aligns with your counters; bump as needed
        self.frame_idx = 0
        self.last_saved_t = -1e9     # for RECORD_FPS throttling

        # QoS setup
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        ctrl_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Subscriptions
        self.create_subscription(Image, IMAGE_TOPIC, self.image_cb, sensor_qos)
        self.create_subscription(Pose, LEFT_POSE_TOPIC, self.left_pose_cb, ctrl_qos)
        self.create_subscription(Pose, RIGHT_POSE_TOPIC, self.right_pose_cb, ctrl_qos)

        # UI / key polling
        self.timer = self.create_timer(1.0 / DISPLAY_FPS, self.ui_step)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        print(
            f"\nSubscribed topics:\n"
            f"  Image: {IMAGE_TOPIC}\n"
            f"  Left pose: {LEFT_POSE_TOPIC}\n"
            f"  Right pose: {RIGHT_POSE_TOPIC}\n"
        )
        print(f"Saving pickles to: {OUTPUT_DIR}")
        print("Controls: [S]=start-record  [E]=end-record  [I]=single image  [C]=cut  [Q]=quit\n")

    # ---------- Callbacks ----------
    def image_cb(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            self.latest_stamp = t
        except Exception as e:
            print(f"[WARN] cv_bridge error: {e}")
            return

        # If we're recording, save frames continuously between S and E
        if self.recording:
            self.maybe_save_record_frame()

    def left_pose_cb(self, msg: Pose):
        self.latest_left_pose = msg

    def right_pose_cb(self, msg: Pose):
        self.latest_right_pose = msg

    # ---------- Recording helpers ----------
    def build_data(self):
        return {
            "image": self.latest_rgb if SAVE_IMAGES else None,
            "left_robot_pos": pose_to_pos_array(self.latest_left_pose),
            "right_robot_pos": pose_to_pos_array(self.latest_right_pose),
            "timestamp": self.latest_stamp,
        }

    def maybe_save_record_frame(self):
        """Save a frame if throttling allows (or if RECORD_FPS=None)."""
        if self.latest_rgb is None:
            return

        t = self.latest_stamp
        if RECORD_FPS is not None:
            min_dt = 1.0 / float(RECORD_FPS)
            if (t - self.last_saved_t) < min_dt:
                return

        data = self.build_data()
        path = os.path.join(OUTPUT_DIR, f"seq_{self.record_id:06d}_frame_{self.frame_idx:06d}.pkl")
        save_to_pickle(data, path)

        self.frame_idx += 1
        self.last_saved_t = t

    def start_recording(self):
        if self.latest_rgb is None:
            print("No image yet; cannot start recording.")
            return
        self.recording = True
        self.frame_idx = 0
        self.last_saved_t = -1e9
        print(f"[REC] START  record_id={self.record_id}  (saving frames between S and E)")

        # Optionally save an explicit "start" marker too
        data = self.build_data()
        path = os.path.join(OUTPUT_DIR, f"start_{self.record_id:06d}.pkl")
        save_to_pickle(data, path)

    def end_recording(self):
        if not self.recording:
            print("[REC] Not recording; ignoring end.")
            return
        self.recording = False
        print(f"[REC] END    record_id={self.record_id}  saved_frames={self.frame_idx}")

        # Save an explicit "end" marker
        if self.latest_rgb is not None:
            data = self.build_data()
            path = os.path.join(OUTPUT_DIR, f"end_{self.record_id:06d}.pkl")
            save_to_pickle(data, path)

        # Advance to next sequence id
        self.record_id += 1

    # ---------- Keyboard control ----------
    def ui_step(self):
        # Display current frame for sanity
        if self.latest_rgb is not None:
            disp = self.latest_rgb.copy()
            status = "REC" if self.recording else "IDLE"
            cv2.putText(
                disp,
                f"{status}  id={self.record_id}  frame={self.frame_idx}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0) if self.recording else (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quit [Q]")
            cv2.destroyAllWindows()
            rclpy.shutdown()
            return

        if key == ord("s"):
            self.start_recording()
        elif key == ord("e"):
            self.end_recording()
        elif key == ord("i"):
            # still allow single snapshots (outside or inside recording)
            if self.latest_rgb is None:
                print("No image yet; not saving.")
                return
            data = self.build_data()
            path = os.path.join(OUTPUT_DIR, f"image_{self.record_id:06d}_{self.frame_idx:06d}.pkl")
            save_to_pickle(data, path)
        elif key == ord("c"):
            if self.latest_rgb is None:
                print("No image yet; not saving.")
                return
            data = self.build_data()
            path = os.path.join(OUTPUT_DIR, f"cut_{self.record_id:06d}_{self.frame_idx:06d}.pkl")
            save_to_pickle(data, path)


def main(argv=None):
    rclpy.init(args=argv)
    node = DataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()