import cv2
import streamlit as st
import numpy as np
import pandas as pd
import mediapipe as mp
import torch

# Streamlit Page Configuration
st.set_page_config(
    page_title="Body Landmark & YOLOv5 Object Detection",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# --- LOAD YOLOV5 MODEL ---
model_weights_path = "./models/best_big_bounding.pt"
# Using torch.hub to load the custom trained YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_weights_path)
model.to("mps")  # Optimized for Mac (Metal Performance Shaders). Use "cuda" for NVIDIA or "cpu"
model.eval()

# --- LOAD IMAGE ---
image_path = "bench2.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for Streamlit/Mediapipe


# Function to detect objects using YOLOv5
def detect_objects(frame):
    results = model(frame)
    pred = results.pred[0] # Get predictions
    return pred


# --- INITIALIZE MEDIAPIPE POSE ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.3, 
    min_tracking_confidence=0.7, 
    model_complexity=2 # Highest quality model for better joint accuracy
)

# Execute YOLOv5 detection on the image
results_yolo = detect_objects(image)

# --- PROCESS YOLOv5 RESULTS ---
if results_yolo is not None:
    for det in results_yolo:
        # Get coordinates of the bounding box
        c1, c2 = det[:2].int(), det[2:4].int()
        cls, conf, *_ = det
        label = f"person {conf:.2f}"

        # Only display if confidence is 70% or higher
        if conf >= 0.7:  
            # Convert tensors to standard tuples
            c1 = (c1[0].item(), c1[1].item())
            c2 = (c2[0].item(), c2[1].item())

            # Extract the 'cropped' frame of the person from the image
            object_frame = image[c1[1] : c2[1], c1[0] : c2[0]]

            # Convert crop to RGB for Pose Estimation processing
            object_frame_rgb = cv2.cvtColor(object_frame, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(object_frame_rgb)

            # Draw landmarks if pose is detected
            if results_pose.pose_landmarks is not None:
                landmarks = results_pose.pose_landmarks.landmark

                # Draw the skeletal connections on the cropped person frame
                for landmark in mp_pose.PoseLandmark:
                    if landmarks[landmark.value].visibility >= 0.3:
                        mp.solutions.drawing_utils.draw_landmarks(
                            object_frame,
                            results_pose.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
                        )

            # Paste the processed crop (with skeletal lines) back into the original image
            image[c1[1] : c2[1], c1[0] : c2[0]] = object_frame

            # Draw the green bounding box and label
            image = cv2.rectangle(image, c1, c2, (0, 255, 0), 2)
            image = cv2.putText(
                image,
                label,
                (c1[0], c1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

# --- DISPLAY OUTPUT ---
st.image(image, caption="YOLOv5 Detection & Pose Estimation Result", use_column_width=True)