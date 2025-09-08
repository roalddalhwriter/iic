"""
AI Posture Coach
----------------
This application is an AI-powered tool that monitors human posture in real-time 
using computer vision and pose estimation. Built with Streamlit and MediaPipe, 
it detects slouching, leaning, and prolonged bad posture, providing immediate 
feedback to the user.  

Key Features:
- Real-time slouch detection based on head–torso ratio
- Lean detection (left/right inclination) based on hip–shoulder alignment
- Calibration mechanism to personalize baseline posture
- Long-term posture risk alerts if poor posture is sustained
- Interactive Streamlit interface with webcam integration

Dependencies:
- streamlit
- streamlit-webrtc
- opencv-python
- mediapipe
- numpy
- threading (for concurrency safety)

Author: [Your Name]
"""

# ===============================
# 📦 Imports
# ===============================
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import threading

# ===============================
# 🎯 MediaPipe Initialization
# ===============================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ===============================
# ⚙️ Streamlit Page Configuration
# ===============================
st.set_page_config(
    page_title="AI Posture Coach",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("AI Posture Coach 🏃‍♂️")

# ===============================
# 🔒 App State Management
# ===============================
# Using a lock to prevent race conditions in multi-threaded environments
lock = threading.Lock()
app_state = {
    "calibrated": False,          # Indicates if calibration is complete
    "baseline_ratio": 0.0,        # Stores baseline slouch ratio after calibration
    "calibration_frames": [],     # Buffer for calibration samples
    "is_calibrating": False       # Flag for calibration process
}

# ===============================
# 🎥 Video Transformer Class
# ===============================
class PostureTransformer(VideoTransformerBase):
    """
    Custom video transformer for Streamlit-WebRTC.
    Processes each video frame to detect posture using MediaPipe landmarks.
    """

    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.bad_posture_counter = 0
        self.BAD_POSTURE_FRAME_THRESHOLD = 200  # ~6–7 seconds of continuous bad posture

    # -------------------------------
    # 🔍 Frame Processing Logic
    # -------------------------------
    def process_frame(self, image):
        """
        Core frame-by-frame processing:
        - Extracts landmarks
        - Calculates lean angle and slouch ratio
        - Applies calibration
        - Generates feedback and warnings
        - Draws overlays on the frame
        """
        h, w, _ = image.shape

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        # Default feedback values
        lean_feedback = "Lean Good"
        slouch_feedback = "Slouch Good"
        is_bad_posture = False

        try:
            # Ensure landmarks are detected
            if not results.pose_landmarks:
                raise Exception("No landmarks detected")

            landmarks = results.pose_landmarks.landmark

            # Draw skeleton on frame
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # -------------------------------
            # 📐 Metric Calculation
            # -------------------------------
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

            # Midpoints
            shoulder_mid_x = int((left_shoulder.x + right_shoulder.x) * w / 2)
            shoulder_mid_y = int((left_shoulder.y + right_shoulder.y) * h / 2)
            hip_mid_x = int((left_hip.x + right_hip.x) * w / 2)
            hip_mid_y = int((left_hip.y + right_hip.y) * h / 2)
            nose_y = int(nose.y * h)

            # Lean Angle
            lean_angle = math.degrees(
                math.atan2(hip_mid_x - shoulder_mid_x, hip_mid_y - shoulder_mid_y)
            )

            # Slouch Ratio
            torso_height = abs(hip_mid_y - shoulder_mid_y)
            neck_height = abs(shoulder_mid_y - nose_y)
            slouch_ratio = neck_height / torso_height if torso_height > 0 else 0

            # -------------------------------
            # 🛠️ Calibration Logic
            # -------------------------------
            with lock:
                if app_state["is_calibrating"]:
                    app_state["calibration_frames"].append(slouch_ratio)
                    if len(app_state["calibration_frames"]) >= 30:
                        app_state["baseline_ratio"] = np.mean(app_state["calibration_frames"])
                        app_state["calibrated"] = True
                        app_state["is_calibrating"] = False
                        app_state["calibration_frames"] = []  # Clear frames
                        st.toast("Calibration Complete!", icon="✅")

            # -------------------------------
            # ✅ Feedback Logic
            # -------------------------------
            if abs(lean_angle) > 7.0:
                lean_feedback = f"LEANING {'LEFT' if lean_angle < 0 else 'RIGHT'}!"
                is_bad_posture = True

            with lock:
                if not app_state["calibrated"]:
                    slouch_feedback = "Please Calibrate"
                else:
                    if (app_state["baseline_ratio"] - slouch_ratio) / app_state["baseline_ratio"] > 0.15:
                        slouch_feedback = "SLOUCHED!"
                        is_bad_posture = True

            # -------------------------------
            # ⏱️ Long-Term Risk Counter
            # -------------------------------
            if is_bad_posture:
                self.bad_posture_counter += 1
            else:
                self.bad_posture_counter = 0

        except Exception:
            # Ignore missing landmarks silently
            pass

        # -------------------------------
        # 🎨 Visualization Overlay
        # -------------------------------
        if 'shoulder_mid_x' in locals():
            cv2.line(image, (shoulder_mid_x, shoulder_mid_y),
                     (hip_mid_x, hip_mid_y), (0, 255, 0), 3)

        # Feedback Text
        cv2.putText(image, f'Lean: {lean_feedback}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f'Slouch: {slouch_feedback}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Calibration Status
        with lock:
            if app_state["is_calibrating"]:
                cv2.putText(image, f'Calibrating... {len(app_state["calibration_frames"])}/30',
                            (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)

        # Long-term risk warning
        if self.bad_posture_counter > self.BAD_POSTURE_FRAME_THRESHOLD:
            warn_text = "RISK: Correct posture to avoid complications!"
            (text_w, text_h), _ = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(image,
                          (w//2 - text_w//2 - 10, h//2 - text_h - 10),
                          (w//2 + text_w//2 + 10, h//2 + 10),
                          (0, 0, 255), -1)
            cv2.putText(image, warn_text, (w//2 - text_w//2, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return image

    # -------------------------------
    # 🔄 Frame Receiver
    # -------------------------------
    def recv(self, frame):
        """Receives frames from webcam, processes them, and returns the modified frame."""
        image = frame.to_ndarray(format="bgr24")
        processed_image = self.process_frame(image)
        return frame.from_ndarray(processed_image, format="bgr24")

# ===============================
# 🖥️ Streamlit UI Layout
# ===============================
with st.sidebar:
    st.header("Settings")
    st.write("Sit in your ideal posture and press the calibrate button to set your baseline.")

    # Calibration Button
    if st.button("Calibrate Good Posture", key="calibrate"):
        with lock:
            if not app_state["is_calibrating"]:
                app_state["is_calibrating"] = True
                app_state["calibration_frames"] = []  # Reset
                app_state["calibrated"] = False

# Webcam Feed Section
st.header("Webcam Feed")
st.write("Click 'START' to begin posture analysis.")

webrtc_streamer(
    key="posture-analysis",
    video_transformer_factory=PostureTransformer,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

# About Section
with st.expander("ℹ️ About the App"):
    st.markdown("""
    This AI-powered application provides real-time feedback on your spine alignment 
    to help you improve your posture.
    
    **Features:**
    - **Slouch Detection:** Identifies forward head posture or hunching. Requires calibration.
    - **Lean Detection:** Detects lateral leaning (scoliosis tendencies).
    - **Long-Term Risk Alerts:** Notifies you if poor posture is maintained for an extended period.
    
    **Instructions:**
    1.  Click **START** to activate your webcam.
    2.  Sit up straight in your best posture.
    3.  Click the **Calibrate Good Posture** button in the sidebar.
    4.  Hold your posture until calibration is complete. The app will then provide live feedback.
    """)
