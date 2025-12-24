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

            # Determine Gaze using Static Landmarks (Corners) to avoid eyelid noise
            
            def get_relative_iris_pos(inner, outer, iris):
                # Vector from Inner to Outer corner (The "X" axis of the eye)
                eye_width_vec = outer - inner
                
                # Vector from Inner corner to Iris
                iris_vec = iris - inner
                
                # Project Iris onto Eye Width line (X-coordinate)
                # Scalar Projection: dot(A, B) / |B|
                # Normalized ratio (0 to 1 along the line): dot(A, B) / |B|^2
                denom = np.dot(eye_width_vec, eye_width_vec)
                if denom == 0: return 0.5, 0.5
                
                x_ratio = np.dot(iris_vec, eye_width_vec) / denom
                
                # For Y, we need a vector perpendicular to the eye width.
                # In screen space (approx), Y is perpendicular to X.
                # Let's rotate eye_width_vec 90 degrees: (dx, dy) -> (-dy, dx)
                # Note: Screen coords, Y increases down.
                # If X is (1, 0), Y perp should be (0, 1).
                # Inner(0,0) -> Outer(10, 0). Vec = (10, 0).
                # Perpendicular = (0, 10).
                # If we rotate 90 deg clockwise: (x, y) -> (-y, x)? No.
                # Let's just use the Orthogonal projection directly.
                # Y_vec = [-eye_width_vec[1], eye_width_vec[0]] # (-y, x) gives 90 deg rotation
                
                # Actually, let's simpler:
                # Vertical displacement relative to the line connecting corners.
                # Cross product 2D (determinant) gives area. Area = Base * Height.
                # Height = Cross / Base.
                
                # Cross Product of (EyeWidth) and (IrisVec).
                # (x1*y2 - y1*x2)
                cross_prod = eye_width_vec[0] * iris_vec[1] - eye_width_vec[1] * iris_vec[0]
                
                # Normalize height by eye width magnitude
                eye_width_len = np.sqrt(denom)
                y_disp = cross_prod / eye_width_len
                
                # Normalize Y score. 
                # This displacement is in pixels. We need a ratio relative to something.
                # Eye height is roughly 1/3 to 1/2 of width?
                # Let's normalize by eye width just to have a unitless ratio.
                # Positive = Below line (Screen Y increases down).
                # Negative = Above line.
                y_ratio = y_disp / eye_width_len 
                
                return x_ratio, y_ratio

            rx, ry = get_relative_iris_pos(right_inner, right_outer, right_iris)
            lx, ly = get_relative_iris_pos(left_inner, left_outer, left_iris)
            
            # Additional Debug Vis
            # Draw eye line
            cv2.line(frame, (int(right_inner[0]), int(right_inner[1])), (int(right_outer[0]), int(right_outer[1])), (255, 0, 0), 1)
            cv2.line(frame, (int(left_inner[0]), int(left_inner[1])), (int(left_outer[0]), int(left_outer[1])), (255, 0, 0), 1)

            avg_x = (rx + lx) / 2.0
            avg_y = (ry + ly) / 2.0

            # Output RAW ratios for Rust to handle calibration
            # avg_x: 0.0 (Left) -> 1.0 (Right) (Theoretical, actual depends on eye physiology)
            # avg_y: 0.0 (Top) -> 1.0 (Bottom)
            
            output = {"x": avg_x, "y": avg_y}
            print(json.dumps(output))
            sys.stdout.flush()
            
            # Debug GUI
            cv2.putText(frame, f"Raw X: {avg_x:.4f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Raw Y: {avg_y:.4f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Iris Tracker Debug", frame)
        if cv2.waitKey(1) & 0xFF == 27: # Esc
            break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
