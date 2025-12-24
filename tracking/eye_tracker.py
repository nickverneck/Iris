import cv2
import mediapipe as mp
import json
import sys
import numpy as np

# Configure MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Camera Setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.stderr.write("Error: Could not open webcam.\n")
    sys.exit(1)

def get_landmark_point(landmarks, idx, w, h):
    point = landmarks[idx]
    return np.array([point.x * w, point.y * h])

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for selfie-view consistency (optional, but good for direct mapping)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # --- Right Eye (User's Right) ---
            # Inner Corner: 33, Outer Corner: 133, Iris Center: 468
            right_inner = get_landmark_point(landmarks, 33, w, h)
            right_outer = get_landmark_point(landmarks, 133, w, h)
            right_iris = get_landmark_point(landmarks, 468, w, h)
            
            # --- Left Eye (User's Left) ---
            # Inner Corner: 362, Outer Corner: 263, Iris Center: 473
            left_inner = get_landmark_point(landmarks, 362, w, h)
            left_outer = get_landmark_point(landmarks, 263, w, h)
            left_iris = get_landmark_point(landmarks, 473, w, h) # 473 is left iris center

            # Calculate relative position (0.0 to 1.0) of iris between corners
            # Simple 1D projection for X:
            # Vector Eye: Outer - Inner
            # Vector Iris: Iris - Inner
            # Projection: Dot product / Length^2
            
            def get_ratio(inner, outer, iris):
                eye_vec = outer - inner
                iris_vec = iris - inner
                eye_len_sq = np.dot(eye_vec, eye_vec)
                if eye_len_sq == 0: return 0.5
                x_ratio = np.dot(iris_vec, eye_vec) / eye_len_sq
                return x_ratio

            # Y Ratio is trickier with just corners. Let's use eyelid top/bottom.
            # Right Eye: Top 159, Bottom 145
            # Left Eye: Top 386, Bottom 374
            
            def get_y_ratio(top, bottom, iris):
                 # Vertical vector
                 eye_h_vec = bottom - top
                 iris_vec = iris - top
                 eye_h_len_sq = np.dot(eye_h_vec, eye_h_vec)
                 if eye_h_len_sq == 0: return 0.5
                 y_ratio = np.dot(iris_vec, eye_h_vec) / eye_h_len_sq
                 return y_ratio

            rx_ratio = get_ratio(right_inner, right_outer, right_iris)
            lx_ratio = get_ratio(left_inner, left_outer, left_iris)
            
            right_top = get_landmark_point(landmarks, 159, w, h)
            right_bottom = get_landmark_point(landmarks, 145, w, h)
            
            left_top = get_landmark_point(landmarks, 386, w, h)
            left_bottom = get_landmark_point(landmarks, 374, w, h)
            
            ry_ratio = get_y_ratio(right_top, right_bottom, right_iris)
            ly_ratio = get_y_ratio(left_top, left_bottom, left_iris)

            # Average the two eyes
            avg_x = (rx_ratio + lx_ratio) / 2.0
            avg_y = (ry_ratio + ly_ratio) / 2.0

            # Normalize/Calibrate broadly (these ratios usually range from 0.3 to 0.7 depending on look)
            # Let's expand them to 0.0-1.0. 
            # Empirically, looking left/right might go 0.2 to 0.8
            # Looking up/down might go 0.2 to 0.8
            
            # Sensitivity factors (user can tune these)
            x_min, x_max = 0.3, 0.7
            y_min, y_max = 0.3, 0.7 # iris moves less vertically
            
            norm_x = (avg_x - x_min) / (x_max - x_min)
            norm_y = (avg_y - y_min) / (y_max - y_min)
            
            norm_x = clamp(norm_x, 0.0, 1.0)
            norm_y = clamp(norm_y, 0.0, 1.0)

            output = {"x": norm_x, "y": norm_y}
            print(json.dumps(output))
            sys.stdout.flush()

except KeyboardInterrupt:
    pass
finally:
    cap.release()
