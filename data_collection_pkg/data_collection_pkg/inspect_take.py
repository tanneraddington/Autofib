import os
import re
import sys
import glob
import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

DATA_ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else "/home/cyndaquil/calibration_data2"
OUT_MP4 = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(DATA_ROOT, os.path.basename(DATA_ROOT) + ".mp4")
FPS = 15.0
TOPIC = "/camera/image_rect"


def num(p):
    m = re.search(r"_(\d+)\.mcap$", p)
    return int(m.group(1)) if m else -1


def to_bgr(msg):
    enc = msg.encoding.lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc == "rgb8":
        return cv2.cvtColor(buf.reshape(msg.height, msg.width, 3), cv2.COLOR_RGB2BGR)
    if enc == "bgr8":
        return buf.reshape(msg.height, msg.width, 3).copy()
    if enc == "mono8":
        return cv2.cvtColor(buf.reshape(msg.height, msg.width), cv2.COLOR_GRAY2BGR)
    raise RuntimeError(f"Unsupported encoding {msg.encoding}")


files = sorted(glob.glob(os.path.join(DATA_ROOT, "*.mcap")), key=num)
if not files:
    sys.exit(f"No .mcap files in {DATA_ROOT}")
print(f"{len(files)} files: {os.path.basename(files[0])} .. {os.path.basename(files[-1])}")

writer, t0, t_last, n = None, None, None, 0

for path in files:
    print("Reading:", os.path.basename(path))
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=path, storage_id="mcap"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if TOPIC not in types:
        del reader
        continue
    Image = get_message(types[TOPIC])
    reader.set_filter(rosbag2_py.StorageFilter(topics=[TOPIC]))

    while reader.has_next():
        _, data, ts = reader.read_next()
        frame = to_bgr(deserialize_message(data, Image))

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(OUT_MP4, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
            if not writer.isOpened():
                sys.exit(f"Could not open video writer: {OUT_MP4}")
            t0 = ts

        cv2.putText(frame, f"t = {(ts - t0) / 1e9:.3f} s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)
        t_last = ts
        n += 1

    del reader

if writer is None:
    sys.exit(f"No {TOPIC} messages found")
writer.release()

dur = (t_last - t0) / 1e9
print(f"Wrote {n} frames to {OUT_MP4}")
if dur > 0:
    print(f"Recording {dur:.1f} s, actual rate {n / dur:.1f} Hz (FPS set to {FPS})")