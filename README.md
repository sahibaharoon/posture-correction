Real-Time AI Exercise Pose Correction
This project is a real-time exercise monitoring system that uses YOLOv5 and Mediapipe to detect human posture during key exercises (bench press, squat, deadlift) and provide live feedback on angles and form.
It also counts repetitions and alerts users with audio and visual feedback when posture is incorrect.
Features
Real-time human detection with YOLOv5
Pose estimation using Mediapipe
Angle calculation for key joints (neck, shoulders, elbows, hips, knees, ankles)
Repetition counting and stage tracking (up/down)
Audio/visual feedback for incorrect posture
Custom ML models for each exercise (pickled classifiers)
Adjustable confidence thresholds
Streamlit-based interactive interface
File Structure
AI_Exercise_Pose_Feedback/
├─ Streamlit.py                 # Main app using YOLOv5 for object detection
├─ Streamlit_NoneYolo.py        # App without YOLO (just Mediapipe pose detection)
├─ requirements.txt             # All dependencies for the project
├─ models/
│  ├─ benchpress/
│  │  └─ benchpress.pkl         # Pickle model for bench press
│  ├─ squat/
│  │  └─ squat.pkl              # Pickle model for squat
│  └─ deadlift/
│     └─ deadlift.pkl           # Pickle model for deadlift
├─ resources/
│  ├─ sounds/                   # Feedback audio files
│  │  ├─ correct.mp3
│  │  ├─ excessive_arch_1.mp3
│  │  └─ ...
│  └─ images/                    # Optional static images or icons
└─ yolov5/                       # Local YOLOv5 repository (for detection)
Dependencies
Make sure you have Python 3.10+. Install the dependencies in a virtual environment:
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
Key dependencies include:
opencv-python – for video capture and frame processing
streamlit – web app interface
mediapipe – pose estimation
torch – YOLOv5 model inference
pygame – audio playback
pandas and numpy – data processing
pickle – load trained ML models
Running the App
1. Run Streamlit with YOLOv5:
# Skip camera authorization prompt (macOS)
OPENCV_AVFOUNDATION_SKIP_AUTH=1 streamlit run Streamlit.py
2. Run Streamlit without YOLOv5 (only Mediapipe):
streamlit run Streamlit_NoneYolo.py
⚠ SSL Fix (macOS Homebrew Python only): If you get
certificate verify failed: unable to get local issuer certificate, run:
pip install --upgrade certifi
export SSL_CERT_FILE=$(python -m certifi)
Using the App
Select the exercise from the sidebar:
Bench Press
Squat
Deadlift
Run the camera (check Run Camera checkbox in Streamlit)
Follow the live pose guidance:
Angles of joints are shown on the sidebar
Audio and visual feedback is given when posture is incorrect
Counter tracks repetitions automatically
Machine Learning Models
Models are pickled classifiers for each exercise.
These take joint positions and visibility as input to predict exercise stage (up/down) and posture correctness.
You can retrain these models using a labeled dataset of joint coordinates if needed.
Notes & Tips
Camera Issues (macOS): If the camera fails to initialize, use:
OPENCV_AVFOUNDATION_SKIP_AUTH=1
Audio Playback: Make sure pygame mixer is installed and your system volume is up.
YOLOv5 weights: Keep best_big_bounding.pt in the models folder.
Virtual environment: Always activate the venv before running Streamlit.
