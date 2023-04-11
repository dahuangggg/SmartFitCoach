import os
import time
from window_capture import capture
import mediapipe as mp
import cv2
import numpy as np
from tqdm import tqdm

from detect_utils import hwnd, landmarks_to_bone_arrays, landmarks_to_numpy, get_bottom_point
from plot_utils import draw_skeleton


def process(pose, img):
    # 对图像进行姿势估计
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # image.flags.writeable = False
    results = pose.process(image)
    # image.flags.writeable = True
    # annotated_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results


if __name__ == "__main__":
    root_path = 'videos/session3'
    filenames = os.listdir(root_path)
    for filename in filenames:
        file_path = os.path.join(root_path, filename)
        cap = cv2.VideoCapture(file_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # print(f"视频总帧数：{frame_count}")
        print(filename)
        # print(f"视频帧率：{fps}")

        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        buffer = []
        t0 = time.time()
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            for i in tqdm(range(frame_count)):
                ret, image = cap.read()
                if not ret:
                    break
                results = process(pose, image)
                if results.pose_landmarks is None:
                    continue
                arr = landmarks_to_numpy(results.pose_landmarks)
                buffer.append(arr)
        t1 = time.time()
        print(f"总共用时{t1 - t0:.2f}s, average fps: {(frame_count / (t1 - t0)):.2f}")
        buffer = np.array(buffer)
        np.save(file_path.replace('.mp4', '.npy'), buffer)
        print("result saved!")

        cap.release()
