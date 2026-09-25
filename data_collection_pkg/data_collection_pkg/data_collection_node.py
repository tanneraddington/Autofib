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
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

# =======================
# ---- USER MACROS ------
# =======================
IMAGE_TOPIC      = "/hy_camera/image"
LEFT_POSE_TOPIC  = "/fwkin_l"
RIGHT_POSE_TOPIC = "/fwkin_r"
STATE_TOPIC      = "/robot/state/current_state"

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

def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


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
        self.latest_state = None

        # Recording state
        self.recording = False
        self.awaiting_label = False
        self.record_id = 100         # aligns with your counters; bump as needed
        self.frame_idx = 0
        self.last_saved_t = -1e9     # for RECORD_FPS throttling

        # Bookkeeping
        self.take_dir = None
        self.t_start = None
        self.t_end = None
        self.duration = None
        self.success = None

        # Continuous streams between start & end
        self.fwkin_l_stream = []
        self.fwkin_r_stream = []
        self.state_stream = []

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
        self.create_subscription(Float32MultiArray, STATE_TOPIC, self.state_cb, ctrl_qos)


        # UI / key polling
        self.timer = self.create_timer(1.0 / DISPLAY_FPS, self.ui_step)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        print(
            f"\nSubscribed topics:\n"
            f"  Image: {IMAGE_TOPIC}\n"
            f"  Left pose: {LEFT_POSE_TOPIC}\n"
            f"  Right pose: {RIGHT_POSE_TOPIC}\n"
            f"  State: {STATE_TOPIC}\n"
        )
        print(f"Saving pickles to: {OUTPUT_DIR}")
        print("Controls: \n"
              "     [S]=start-record  [E]=end-record\n"  
              "     [Y]=label success  [N]=label failure\n"  
              "     [Q]=quit\n")

    # ---------- Callbacks ----------
    def image_cb(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            #t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            self.latest_stamp = stamp_to_sec(msg.header.stamp)
        except Exception as e:
            print(f"[WARN] cv_bridge error: {e}")
            return

        # If we're recording, save frames continuously between S and E
        if self.recording:
            self.maybe_save_record_frame()

    def left_pose_cb(self, msg: Pose):
        self.latest_left_pose = msg
        if self.recording:
            t = self.get_clock().now().nanoseconds*1e-9
            self.fwkin_l_stream.append({
                "t": float(t),
                "pos": pose_to_pos_array(msg),
                #"raw": msg,
            })

    def right_pose_cb(self, msg: Pose):
        self.latest_right_pose = msg
        if self.recording:
            t = self.get_clock().now().nanoseconds * 1e-9
            self.fwkin_r_stream.append({
                "t": float(t),
                "pos": pose_to_pos_array(msg),
                #"raw": msg,
            })
    
    def state_cb(self, msg):
        self.latest_state = msg
        if self.recording:
            t = self.get_clock().now().nanoseconds * 1e-9
            self.state_stream.append({
                "t": float(t),
                "data": list(msg.data),
            })

    # ---------- Recording helpers ----------
    def build_frame_dict(self):
        return {
            "image": self.latest_rgb if SAVE_IMAGES else None,
            "timestamp": self.latest_stamp,
            # Keep snapshot of latest frame too
            "left_robot_pos_latest": pose_to_pos_array(self.latest_left_pose),
            "right_robot_pos_latest": pose_to_pos_array(self.latest_right_pose),
            "robot_state_latest": self.latest_state.data if self.latest_state is not None else None,
        }

    def maybe_save_record_frame(self):
        """Save a frame if throttling allows (or if RECORD_FPS=None)."""
        if self.latest_rgb is None or self.take_dir is None:
            return

        t = self.latest_stamp
        if RECORD_FPS is not None:
            min_dt = 1.0 / float(RECORD_FPS)
            if (t - self.last_saved_t) < min_dt:
                return

        data = self.build_frame_dict()
        path = os.path.join(self.take_dir, f"frame_{self.frame_idx:06d}.pkl")
        save_to_pickle(data, path)

        self.frame_idx += 1
        self.last_saved_t = t

    def start_recording(self):
        if self.latest_rgb is None:
            print("No image yet; cannot start recording.")
            return
        if self.recording:
            print("Already recording.")
            return
        self.recording = True
        self.awaiting_label = False
        self.success = None

        self.frame_idx = 0
        self.last_saved_t = -1e9

        self.t_start = self.latest_stamp
        self.t_end = None
        self.duration = None

        self.fwkin_l_stream = []
        self.fwkin_r_stream = []
        self.state_stream = []

        self.take_dir = os.path.join(OUTPUT_DIR, f"take_{self.record_id:06d}")
        os.makedirs(self.take_dir,exist_ok=True)

        print(f"[REC] START record_id={self.record_id} -> {self.take_dir}")

        save_to_pickle(self.build_frame_dict(),os.path.join(self.take_dir,"start.pkl"))

    def end_recording(self):
        if not self.recording:
            print("[REC] Not recording; ignoring end.")
            return
        self.recording = False
        self.t_end = self.latest_stamp
        self.duration = (self.t_end - self.t_start) if (self.t_start is not None and self.t_end is not None) else None

        print(f"[REC] END    record_id={self.record_id}  saved_frames={self.frame_idx}")

        if self.duration is not None:
            print(f"[REC] Duration: {self.duration:.3f} sec")
        print(f"[REC] Streams: fwkin_l={len(self.fwkin_l_stream)} fwkin_r={len(self.fwkin_r_stream)} state={len(self.state_stream)}")
        print("[LABEL] Press Y for success for N for failure...")

        # Save an explicit "end" marker
        if self.latest_rgb is not None and self.take_dir is not None:
            save_to_pickle(self.build_frame_dict(), os.path.join(self.take_dir, "end.pkl"))

        self.awaiting_label = True

    def finalize_take(self,success: bool):
        self.success = bool(success)
        self.awaiting_label = False

        take = {
            "record_id": self.record_id,
            "topics": {
                "image": IMAGE_TOPIC,
                "fwkin_l": LEFT_POSE_TOPIC,
                "fwkin_r": RIGHT_POSE_TOPIC,
                "robot_state": STATE_TOPIC,
            },
            "t_start_image": self.t_start,
            "t_end_image": self.t_end,
            "duration_sec": self.duration,
            "saved_image_frames": self.frame_idx,
            "record_fps_images": RECORD_FPS,
            "save_images": SAVE_IMAGES,
            "user_success": self.success,

            # Continuous streams between S and E
            "fwkin_l_stream": self.fwkin_l_stream,
            "fwkin_r_stream": self.fwkin_r_stream,
            "robot_state_stream": self.state_stream,

            "take_dir": self.take_dir,
            "frame_files_pattern": "frame_######.pkl",
        }

        if self.take_dir is not None:
            save_to_pickle(take, os.path.join(self.take_dir,"take.pkl"))

        print(f"[DONE] record_id={self.record_id} labeled success={self.success}")

        # Advance and reset
        self.record_id += 1
        self.take_dir = None
        self.t_start = None
        self.t_end = None
        self.duration = None
        self.frame_idx = 0
        self.fwkin_l_stream = []
        self.fwkin_r_stream = []
        self.state_stream = []

    # ---------- Keyboard control ----------
    def ui_step(self):
        # Display current frame for sanity
        if self.latest_rgb is not None:
            disp = self.latest_rgb.copy()
            if self.recording:
                status = "REC"
            elif self.awaiting_label:
                status = "LABEL (Y/N)"
            else:
                status = "IDLE"

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
            #cv2.imshow(WINDOW_NAME, cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))

            if self.recording:
                cv2.putText(
                        disp,
                        f"L:{len(self.fwkin_l_stream)} R:{len(self.fwkin_r_stream)} S:{len(self.state_stream)}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
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
        
        if self.awaiting_label:
            if key == ord("y"):
                self.finalize_take(True)
            elif key == ord("n"):
                self.finalize_take(False)
            return
        

        if key == ord("s"):
            self.start_recording()
        elif key == ord("e"):
            self.end_recording()


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