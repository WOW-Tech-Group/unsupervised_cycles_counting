import os
import cv2
import shutil
import numpy as np


def get_framename(video_path) :
    file_name = video_path.split('/')[-1]
    raw_name = file_name.split('.')[0]
    return raw_name


def get_optical_flow(gray, prev_gray, mask):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale=0.5, levels=5, winsize=11,
                                        iterations=5, poly_n=5, poly_sigma=1.1, flags=0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Set image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 1] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return mask


def save_frames(video_path, frames_save_path, flow_save_path, size):
    cap = cv2.VideoCapture(video_path)
    not_end_of_video, current_frame = cap.read()
    current_frame = cv2.resize(current_frame, size)
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGRA2GRAY)

    mask = np.empty_like(current_frame)
    frame_nb = 0

    if not os.path.isdir(frames_save_path):
        os.mkdir(frames_save_path)
    if not os.path.isdir(flow_save_path):
        os.mkdir(flow_save_path)

    while not_end_of_video :
        frame_path = os.path.join(frames_save_path, str(frame_nb) + '.jpg')
        cv2.imwrite(frame_path, current_frame)
        prev_gray = gray[:]
        not_end_of_video, current_frame = cap.read()
        if not_end_of_video :
            current_frame = cv2.resize(current_frame, size)
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGRA2GRAY)
            flow = get_optical_flow(gray, prev_gray, mask)
            flow_path = os.path.join(flow_save_path, str(frame_nb) + '.jpg')
            cv2.imwrite(flow_path, flow)
        frame_nb += 1


def extract_frames_and_flow(video_path, force_reset, size=(256, 256)):
    frames_and_flow_name = get_framename(video_path)
    frames_and_flow_path = os.path.join("./", frames_and_flow_name)
    if not os.path.exists(frames_and_flow_path) or force_reset:
        if os.path.exists(frames_and_flow_path) :
            shutil.rmtree(frames_and_flow_path)
        else :
            print("Extracting the video frames and optical flow...")
        os.mkdir(frames_and_flow_path)
        frames_path = os.path.join(frames_and_flow_path, "frames")
        flow_path = os.path.join(frames_and_flow_path, "flow")
        save_frames(video_path, frames_path, flow_path, size)
        print("Extraction completed.")
    else :
        print("Pre-existing video frames and flow extraction.")
    return frames_and_flow_path