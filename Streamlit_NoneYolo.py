import cv2
import streamlit as st
import numpy as np
import pandas as pd
import mediapipe as mp
import datetime
import time
import pygame
import torch
import pickle
import random

st.set_page_config(
    page_title="Real-Time AI Posture Correction for 3 Major Exercises",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

previous_alert_time = 0
pygame.mixer.init()

def most_frequent(data):
    return max(data, key=data.count)

def calculateAngle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

st.title("Real-Time AI Posture Correction for 3 Major Exercises")

menu_selection = st.selectbox("Select Exercise", ("Bench Press", "Squat", "Deadlift"))
counter_display = st.sidebar.empty()
counter = 0
counter_display.header(f"Current Count: {counter} reps")
current_stage = ""
posture_status = [None]

# Load model
if menu_selection == "Bench Press":
    model_path = "./models/benchpress/benchpress.pkl"
elif menu_selection == "Squat":
    model_path = "./models/squat/squat.pkl"
else:
    model_path = "./models/deadlift/deadlift.pkl"

with open(model_path, "rb") as f:
    model_e = pickle.load(f)

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.7, model_complexity=2)
confidence_threshold = st.sidebar.slider("Landmark Detection Confidence Threshold", 0.0, 1.0, 0.7)

# Angle placeholders
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

while True:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)

    object_frame = frame
    results_pose = pose.process(object_frame)

    if results_pose.pose_landmarks:
        landmarks = results_pose.pose_landmarks.landmark
        # Extract all landmark points (same as previous code)
        # Calculate angles (same as previous code)
        # Display angles (English labels)
        # Count repetitions (same logic)
        # Play audio using pygame for feedback
        # Draw landmarks
        for landmark in mp_pose.PoseLandmark:
            if landmarks[landmark.value].visibility >= confidence_threshold:
                mp.solutions.drawing_utils.draw_landmarks(
                    object_frame,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                )

    FRAME_WINDOW.image(object_frame)