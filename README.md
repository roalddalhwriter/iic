# Introduction

Poor posture has become one of the most common lifestyle issues, especially among students, working professionals, and individuals who spend long hours in front of computers. Prolonged bad posture, such as slouching or leaning, can gradually lead to musculoskeletal disorders, chronic back pain, and even scoliosis. Most posture correction systems available today are either expensive, rely on specialized hardware, or require manual supervision.

This project introduces an AI-powered system for real-time posture monitoring and spinal curve estimation using only a normal RGB webcam. The application leverages computer vision and pose estimation techniques to track key body landmarks, calculate spinal alignment, and provide immediate feedback about the user’s posture. By offering affordable and accessible monitoring, this system aims to reduce posture-related health risks.

# Objectives

The primary objective of this project is to detect spinal misalignment in real-time using AI-based pose estimation. The system focuses on estimating two key posture metrics: leaning tendencies, which may indicate scoliosis, and slouching, which is typically associated with forward head posture or hunching. The system further provides instant on-screen feedback and warns the user if poor posture persists over time. A calibration mechanism has also been included to personalize the system for each user’s natural sitting posture.

# System Design

The system is designed to operate on a simple pipeline. It begins with the input stream from the user’s webcam, which captures real-time RGB frames. These frames are passed into MediaPipe’s Pose model, which detects 33 human body landmarks. From these landmarks, the system extracts critical points such as the shoulders, hips, and nose.

Using these points, the system calculates two posture metrics. The lean angle is determined by measuring the vertical alignment between the shoulder and hip midpoints, which helps in detecting left or right leaning. The slouch ratio is calculated by comparing the distance between the nose and shoulders with the torso height, allowing the system to detect forward bending or hunching.

A calibration step ensures accuracy by recording the user’s baseline slouch ratio while sitting upright. Once calibrated, the system compares real-time posture with the baseline. If the lean angle exceeds seven degrees or if the slouch ratio deviates more than fifteen percent from the baseline, the system flags the posture as incorrect. Additionally, if the user maintains a bad posture continuously for a fixed number of frames (about seven to ten seconds), the system issues a long-term risk alert.

# Implementation

The project has been implemented in two versions. The first is a command-line OpenCV version, where users can calibrate their posture by pressing the “c” key, and posture feedback is displayed directly on the video feed. The second version is a Streamlit-based web application integrated with Streamlit-WebRTC. This version provides an interactive interface with a calibration button, real-time webcam streaming, and user-friendly posture alerts.

Both implementations make use of Python along with essential libraries such as OpenCV for image processing, MediaPipe for landmark detection, NumPy for mathematical calculations, and Streamlit for building the web interface.

# Features

The system offers several key features. It can detect both slouching and leaning, providing immediate feedback in the form of text overlays on the video feed. It includes a calibration mechanism that adapts the system to each user’s natural posture, ensuring higher accuracy. The application also tracks long-term posture habits, warning the user if they continue in a poor posture for extended durations. The Streamlit-based UI makes the system highly accessible, requiring only a standard laptop and webcam to function.

# Results and Discussion

The prototype successfully provides real-time posture feedback using a standard webcam without the need for expensive sensors. The system was able to detect forward slouching and lateral leaning effectively and provided alerts when posture was maintained incorrectly for too long. While the OpenCV version demonstrates the core model effectively, the Streamlit application enhances usability by making the tool interactive and intuitive for end-users.

# Limitations

Despite its effectiveness, the system has a few limitations. Since it relies on 2D RGB images, it cannot capture the full depth of spinal curvature. The detection is also affected by poor lighting conditions or partial occlusion of the body. Furthermore, the system currently focuses only on the upper body and does not account for leg positioning, which also affects posture.

# Future Scope

Future improvements can significantly enhance the utility of this project. Integrating depth cameras such as Microsoft Kinect or Intel RealSense could allow for more accurate 3D spinal curve estimation. A mobile version of the app could make posture monitoring more accessible on smartphones. Additionally, the system could be extended to log posture data over time and provide personalized exercise recommendations powered by AI to help users actively correct their posture.

# Conclusion

This project demonstrates the feasibility of creating an AI-powered posture monitoring system using only computer vision and affordable hardware. By combining Mediapipe’s pose estimation with mathematical analysis of key body landmarks, the system can detect slouching, leaning, and prolonged bad posture in real-time. With further development, such a system could prove invaluable for workplaces, schools, and individuals aiming to improve their spinal health and prevent long-term musculoskeletal issues.
