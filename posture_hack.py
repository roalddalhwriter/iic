import cv2
import mediapipe as mp
import numpy as np
import math

# --- Initialize MediaPipe ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Calibration Variables ---
calibrated = False
baseline_ratio = 0.0
calibration_frames = []
CALIBRATION_SET_SIZE = 30 

# --- NEW: Long-Term Risk Variables ---
bad_posture_counter = 0  # Counts consecutive frames of bad posture
BAD_POSTURE_FRAME_THRESHOLD = 300  # Approx. 10 seconds at 30fps. Increase this for real use.

# --- Start Webcam ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    key = cv2.waitKey(5) & 0xFF # Get keypress

    try:
        if not results.pose_landmarks:
            raise Exception("No landmarks detected")
            
        # Draw the FULL BODY skeleton
        mp_drawing.draw_landmarks(
            image_bgr,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        landmarks = results.pose_landmarks.landmark
        
        # Get Coordinates
        left_shoulder_pt = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder_pt = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip_pt = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip_pt = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        nose_pt = landmarks[mp_pose.PoseLandmark.NOSE.value]

        shoulder_mid_x = int((left_shoulder_pt.x + right_shoulder_pt.x) * w / 2)
        shoulder_mid_y = int((left_shoulder_pt.y + right_shoulder_pt.y) * h / 2)
        hip_mid_x = int((left_hip_pt.x + right_hip_pt.x) * w / 2)
        hip_mid_y = int((left_hip_pt.y + right_hip_pt.y) * h / 2)
        nose_y = int(nose_pt.y * h)

        # === METRIC 1: LEAN ANGLE (Scoliosis Tendency) ===
        delta_x = hip_mid_x - shoulder_mid_x
        delta_y = hip_mid_y - shoulder_mid_y
        lean_angle = math.degrees(math.atan2(delta_x, delta_y))

        # === METRIC 2: SLOUCH RATIO ===
        try:
            torso_height = abs(hip_mid_y - shoulder_mid_y)
            neck_height = abs(shoulder_mid_y - nose_y)
            slouch_ratio = neck_height / torso_height
        except ZeroDivisionError:
            slouch_ratio = 0.0 
            
        # === Calibration Logic ===
        if key == ord('c') and not calibrated:
            print("Calibrating... Stay still in your GOOD posture.")
            calibration_frames.append(slouch_ratio)
            
        if len(calibration_frames) >= CALIBRATION_SET_SIZE:
            baseline_ratio = sum(calibration_frames) / len(calibration_frames)
            calibrated = True
            calibration_frames = [] 
            print(f"Calibration Complete! Baseline ratio set to: {baseline_ratio:.4f}")

        # === Feedback Logic ===
        lean_feedback = ""
        slouch_feedback = ""
        is_bad_posture = False # Flag for our timer
        
        lean_threshold = 7.0 
        if abs(lean_angle) > lean_threshold:
            lean_feedback = f"LEANING {'LEFT' if lean_angle > 0 else 'RIGHT'}!"
            is_bad_posture = True # THIS IS A BAD POSTURE
        else:
            lean_feedback = "Lean Good"

        if not calibrated:
            slouch_feedback = "Press 'c' to Calibrate"
        else:
            slouch_threshold_percentage = 0.15 
            current_deviation = (baseline_ratio - slouch_ratio) / baseline_ratio
            
            if current_deviation > slouch_threshold_percentage:
                slouch_feedback = "SLOUCHED!"
                is_bad_posture = True # THIS IS ALSO A BAD POSTURE
            else:
                slouch_feedback = "Slouch Good"

        # === NEW: Long-Term Risk Logic ===
        if is_bad_posture:
            bad_posture_counter += 1 # Increment counter every frame the posture is bad
        else:
            bad_posture_counter = 0 # Reset counter if posture is good

        # === Visualization ===
        # Draw our custom posture line
        cv2.line(image_bgr, (shoulder_mid_x, shoulder_mid_y), (hip_mid_x, hip_mid_y), (0, 255, 0), 3) 
        
        # Display feedback text
        cv2.putText(image_bgr, f'Lean Angle: {lean_angle:.2f} deg ({lean_feedback})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image_bgr, f'Slouch Ratio: {slouch_ratio:.2f} ({slouch_feedback})', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display calibration progress
        if not calibrated and len(calibration_frames) > 0:
            cv2.putText(image_bgr, f'Calibrating... {len(calibration_frames)}/{CALIBRATION_SET_SIZE}', (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display LONG-TERM RISK WARNING
        if bad_posture_counter > BAD_POSTURE_FRAME_THRESHOLD:
            # Display a big warning message in the middle of the screen
            warn_text = "RISK: Correct posture to avoid complications!"
            text_size = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2 + 50 # Position it below center
            
            cv2.rectangle(image_bgr, (text_x - 10, text_y - text_size[1] - 10), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 255), -1) # Red background
            cv2.putText(image_bgr, warn_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    except Exception as e:
        pass 

    cv2.imshow('AI Posture Coach', image_bgr)

    if key == 27: # Exit on 'ESC'
        break

cap.release()
cv2.destroyAllWindows()
pose.close()