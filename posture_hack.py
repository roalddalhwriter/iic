import cv2
import mediapipe as mp
import numpy as np
import math

# ================================
# Initialization
# ================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Calibration Parameters
CALIBRATION_SET_SIZE = 30
calibrated = False
baseline_ratio = 0.0
calibration_frames = []

# Long-Term Risk Parameters
bad_posture_counter = 0
BAD_POSTURE_FRAME_THRESHOLD = 300  # ~10 seconds at 30fps


# ================================
# Helper Functions
# ================================
def calculate_metrics(landmarks, w, h):
    """Calculate lean angle and slouch ratio using pose landmarks."""
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

    # Lean angle
    delta_x = hip_mid_x - shoulder_mid_x
    delta_y = hip_mid_y - shoulder_mid_y
    lean_angle = math.degrees(math.atan2(delta_x, delta_y))

    # Slouch ratio
    torso_height = abs(hip_mid_y - shoulder_mid_y)
    neck_height = abs(shoulder_mid_y - nose_y)
    slouch_ratio = neck_height / torso_height if torso_height > 0 else 0

    return lean_angle, slouch_ratio, (shoulder_mid_x, shoulder_mid_y), (hip_mid_x, hip_mid_y)


def give_feedback(lean_angle, slouch_ratio, calibrated, baseline_ratio):
    """Generate posture feedback and check for bad posture."""
    lean_feedback = "Lean Good"
    slouch_feedback = "Slouch Good"
    is_bad_posture = False

    # Lean check
    if abs(lean_angle) > 7.0:
        lean_feedback = f"LEANING {'LEFT' if lean_angle > 0 else 'RIGHT'}!"
        is_bad_posture = True

    # Slouch check
    if not calibrated:
        slouch_feedback = "Press 'c' to Calibrate"
    else:
        deviation = (baseline_ratio - slouch_ratio) / baseline_ratio
        if deviation > 0.15:
            slouch_feedback = "SLOUCHED!"
            is_bad_posture = True

    return lean_feedback, slouch_feedback, is_bad_posture


# ================================
# Main Loop
# ================================
def main():
    global calibrated, baseline_ratio, calibration_frames, bad_posture_counter

    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        key = cv2.waitKey(5) & 0xFF

        try:
            if not results.pose_landmarks:
                raise Exception("No landmarks detected")

            # Draw skeleton
            mp_drawing.draw_landmarks(
                image_bgr,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # Calculate metrics
            landmarks = results.pose_landmarks.landmark
            lean_angle, slouch_ratio, shoulder_mid, hip_mid = calculate_metrics(landmarks, w, h)

            # Calibration
            if key == ord('c') and not calibrated:
                print("Calibrating... Stay still in your GOOD posture.")
                calibration_frames.append(slouch_ratio)

            if len(calibration_frames) >= CALIBRATION_SET_SIZE:
                baseline_ratio = np.mean(calibration_frames)
                calibrated = True
                calibration_frames = []
                print(f"Calibration Complete! Baseline ratio = {baseline_ratio:.4f}")

            # Feedback
            lean_feedback, slouch_feedback, is_bad_posture = give_feedback(
                lean_angle, slouch_ratio, calibrated, baseline_ratio
            )

            # Long-term risk counter
            if is_bad_posture:
                bad_posture_counter += 1
            else:
                bad_posture_counter = 0

            # Visualization
            cv2.line(image_bgr, shoulder_mid, hip_mid, (0, 255, 0), 3)
            cv2.putText(image_bgr, f'Lean: {lean_angle:.2f}° ({lean_feedback})',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image_bgr, f'Slouch: {slouch_ratio:.2f} ({slouch_feedback})',
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Calibration progress
            if not calibrated and calibration_frames:
                cv2.putText(image_bgr,
                            f'Calibrating... {len(calibration_frames)}/{CALIBRATION_SET_SIZE}',
                            (w // 2 - 150, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Long-term warning
            if bad_posture_counter > BAD_POSTURE_FRAME_THRESHOLD:
                warn_text = "RISK: Correct posture to avoid complications!"
                (text_w, text_h), _ = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.rectangle(image_bgr,
                              (w // 2 - text_w // 2 - 10, h // 2 - text_h - 10),
                              (w // 2 + text_w // 2 + 10, h // 2 + 10),
                              (0, 0, 255), -1)
                cv2.putText(image_bgr, warn_text,
                            (w // 2 - text_w // 2, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        except Exception:
            pass

        cv2.imshow('AI Posture Coach', image_bgr)

        if key == 27:  # ESC to quit
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    pose.close()


# ================================
# Entry Point
# ================================
if __name__ == "__main__":
    main()
