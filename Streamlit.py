import cv2
import streamlit as st
import numpy as np
import pandas as pd
import mediapipe as mp
import datetime
import time
import random
import torch
import pickle
import pathlib

st.set_page_config(
    page_title="Real-Time AI Posture Correction for 3 Major Exercises",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# YOLOv5 model load
ROOT = pathlib.Path(__file__).resolve().parent
model_weights_path = ROOT / "models" / "best_big_bounding.pt"

model = torch.hub.load(
    str(ROOT / "yolov5"),   # local yolov5 repo
    "custom",
    path=str(model_weights_path),
    source="local"
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval()

previous_alert_time = 0

def most_frequent(data):
    return max(data, key=data.count)

# Angle calculation
def calculateAngle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Object detection using YOLOv5
def detect_objects(frame):
    results = model(frame)
    pred = results.pred[0]
    return pred

st.title("Real-Time AI Posture Correction for 3 Major Exercises")

# Sidebar menu
menu_selection = st.selectbox("Select Exercise", ("Bench Press", "Squat", "Deadlift"))
counter_display = st.sidebar.empty()
counter = 0
counter_display.header(f"Current Count: {counter} reps")
current_stage = ""
posture_status = [None]

# Load exercise-specific model
if menu_selection == "Bench Press":
    model_path = "./models/benchpress/benchpress.pkl"
elif menu_selection == "Squat":
    model_path = "./models/squat/squat.pkl"
else:
    model_path = "./models/deadlift/deadlift.pkl"

with open(model_path, "rb") as f:
    model_e = pickle.load(f)

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# Mediapipe Pose initialization
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.7, model_complexity=2)

confidence_threshold = st.sidebar.slider("Landmark Detection Confidence Threshold", 0.0, 1.0, 0.7)

# Angle display placeholders
neck_angle_display = st.sidebar.empty()
left_shoulder_angle_display = st.sidebar.empty()
right_shoulder_angle_display = st.sidebar.empty()
left_elbow_angle_display = st.sidebar.empty()
right_elbow_angle_display = st.sidebar.empty()
left_hip_angle_display = st.sidebar.empty()
right_hip_angle_display = st.sidebar.empty()
left_knee_angle_display = st.sidebar.empty()
right_knee_angle_display = st.sidebar.empty()
left_ankle_angle_display = st.sidebar.empty()
right_ankle_angle_display = st.sidebar.empty()

run = st.checkbox("Run Camera")
while True:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)

    # YOLOv5 detection
    results_yolo = detect_objects(frame)

    try:
        if results_yolo is not None:
            for det in results_yolo:
                c1, c2 = det[:2].int(), det[2:4].int()
                cls, conf, *_ = det
                if conf >= 0.7:
                    c1 = (c1[0].item(), c1[1].item())
                    c2 = (c2[0].item(), c2[1].item())
                    object_frame = frame[c1[1]:c2[1], c1[0]:c2[0]]
                    results_pose = pose.process(object_frame)

                    if results_pose.pose_landmarks:
                        landmarks = results_pose.pose_landmarks.landmark
                        nose = [landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y]
                        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
                        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
                        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
                        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
                        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
                        left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y]
                        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
                        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
                        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
                        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
                        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
                        right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y]

                        # Calculate angles
                        neck_angle = (calculateAngle(left_shoulder, nose, left_hip) + calculateAngle(right_shoulder, nose, right_hip))/2
                        left_elbow_angle = calculateAngle(left_shoulder, left_elbow, left_wrist)
                        right_elbow_angle = calculateAngle(right_shoulder, right_elbow, right_wrist)
                        left_shoulder_angle = calculateAngle(left_elbow, left_shoulder, left_hip)
                        right_shoulder_angle = calculateAngle(right_elbow, right_shoulder, right_hip)
                        left_hip_angle = calculateAngle(left_shoulder, left_hip, left_knee)
                        right_hip_angle = calculateAngle(right_shoulder, right_hip, right_knee)
                        left_knee_angle = calculateAngle(left_hip, left_knee, left_ankle)
                        right_knee_angle = calculateAngle(right_hip, right_knee, right_ankle)
                        left_ankle_angle = calculateAngle(left_knee, left_ankle, left_heel)
                        right_ankle_angle = calculateAngle(right_knee, right_ankle, right_heel)

                        # Display angles
                        neck_angle_display.text(f"Neck Angle: {neck_angle:.2f}°")
                        left_shoulder_angle_display.text(f"Left Shoulder Angle: {left_shoulder_angle:.2f}°")
                        right_shoulder_angle_display.text(f"Right Shoulder Angle: {right_shoulder_angle:.2f}°")
                        left_elbow_angle_display.text(f"Left Elbow Angle: {left_elbow_angle:.2f}°")
                        right_elbow_angle_display.text(f"Right Elbow Angle: {right_elbow_angle:.2f}°")
                        left_hip_angle_display.text(f"Left Hip Angle: {left_hip_angle:.2f}°")
                        right_hip_angle_display.text(f"Right Hip Angle: {right_hip_angle:.2f}°")
                        left_knee_angle_display.text(f"Left Knee Angle: {left_knee_angle:.2f}°")
                        right_knee_angle_display.text(f"Right Knee Angle: {right_knee_angle:.2f}°")
                        left_ankle_angle_display.text(f"Left Ankle Angle: {left_ankle_angle:.2f}°")
                        right_ankle_angle_display.text(f"Right Ankle Angle: {right_ankle_angle:.2f}°")

                        # Count repetitions
                        try:
                            row = [coord for res in results_pose.pose_landmarks.landmark for coord in [res.x, res.y, res.z, res.visibility]]
                            X = pd.DataFrame([row])
                            exercise_class = model_e.predict(X)[0]
                            exercise_class_prob = model_e.predict_proba(X)[0]

                            if "down" in exercise_class:
                                current_stage = "down"
                                posture_status.append(exercise_class)
                            elif current_stage=="down" and "up" in exercise_class:
                                current_stage = "up"
                                counter += 1
                                posture_status.append(exercise_class)
                                counter_display.header(f"Current Count: {counter} reps")
                        except Exception:
                            pass

                        # Draw landmarks
                        for landmark in mp_pose.PoseLandmark:
                            if landmarks[landmark.value].visibility >= confidence_threshold:
                                mp.solutions.drawing_utils.draw_landmarks(
                                    object_frame,
                                    results_pose.pose_landmarks,
                                    mp_pose.POSE_CONNECTIONS,
                                    mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                                )

                frame = object_frame

        FRAME_WINDOW.image(frame)

    except Exception:
        pass