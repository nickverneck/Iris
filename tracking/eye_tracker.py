import cv2
import mediapipe as mp
import json
import sys
import numpy as np

import cv2
import mediapipe as mp
import json
import sys
import numpy as np
import select

# ... (Previous imports and FaceMesh setup remain same)
# Configure MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
# Start in full screen for calibration visibility
cv2.namedWindow("Iris Tracker", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Iris Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

calibration_mode = False
calib_dot = None # (x, y) relative 0-1

def get_landmark_point(landmarks, idx, w, h):
    point = landmarks[idx]
    return np.array([point.x * w, point.y * h])
    
def get_relative_iris_pos(inner, outer, iris):
    eye_width_vec = outer - inner
    iris_vec = iris - inner
    denom = np.dot(eye_width_vec, eye_width_vec)
    if denom == 0: return 0.5, 0.5
    x_ratio = np.dot(iris_vec, eye_width_vec) / denom
    
    cross_prod = eye_width_vec[0] * iris_vec[1] - eye_width_vec[1] * iris_vec[0]
    eye_width_len = np.sqrt(denom)
    y_ratio = cross_prod / eye_width_len / eye_width_len 
    return x_ratio, y_ratio

# ...
screen_w, screen_h = 1920, 1080 # Default fallback
canvas = None

try:
    while True:
        # 1. Check for Input Commands (Non-blocking)
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if line:
                try:
                    cmd = json.loads(line)
                    if cmd.get("type") == "init":
                        screen_w = int(cmd["width"])
                        screen_h = int(cmd["height"])
                        # Resize window to match screen
                        cv2.resizeWindow("Iris Tracker", screen_w, screen_h)
                        
                    elif cmd.get("type") == "calibration_point":
                        calibration_mode = True
                        calib_dot = (cmd["x"], cmd["y"])
                    elif cmd.get("type") == "calibration_end":
                        calibration_mode = False
                        calib_dot = None
                except:
                    pass

        # 2. Capture & Process
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Prepare Display
        if calibration_mode:
            # Create black canvas matching screen size
            if canvas is None or canvas.shape[1] != screen_w or canvas.shape[0] != screen_h:
                canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            else:
                canvas.fill(0) # Clear
            
            # Draw Webcam Feed small in bottom right or center-faded?
            # Let's put it in the background but darkened?
            # Scaling camera to screen while preserving aspect ratio is complex.
            # Simpler: Just overlay camera in top-right corner so user can check eye tracking.
            cam_small_w = int(screen_w * 0.2)
            cam_small_h = int(cam_small_w * (h / w))
            cam_small = cv2.resize(frame, (cam_small_w, cam_small_h))
            
            # Overlay
            canvas[0:cam_small_h, screen_w-cam_small_w:screen_w] = cam_small
            
            # Draw Calibration Dot
            if calib_dot:
                cx = int(calib_dot[0] * screen_w)
                cy = int(calib_dot[1] * screen_h)
                cv2.circle(canvas, (cx, cy), 30, (0, 0, 255), -1) 
                cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), -1)
                cv2.putText(canvas, "Look at the DOT and press SPACE", (cx - 150, cy + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            final_display = canvas
        else:
            final_display = frame

        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # ... (Landmark extraction code same as before) ...
            # --- Right Eye ---
            right_inner = get_landmark_point(landmarks, 33, w, h)
            right_outer = get_landmark_point(landmarks, 133, w, h)
            right_iris = get_landmark_point(landmarks, 468, w, h)
            
            # --- Left Eye ---
            left_inner = get_landmark_point(landmarks, 362, w, h)
            left_outer = get_landmark_point(landmarks, 263, w, h)
            left_iris = get_landmark_point(landmarks, 473, w, h)

            rx, ry = get_relative_iris_pos(right_inner, right_outer, right_iris)
            lx, ly = get_relative_iris_pos(left_inner, left_outer, left_iris)
            
            # Debug Vis on Camera Frame (if visible)
            if not calibration_mode:
                cv2.circle(frame, (int(right_iris[0]), int(right_iris[1])), 2, (0, 255, 0), -1)
                cv2.circle(frame, (int(left_iris[0]), int(left_iris[1])), 2, (0, 255, 0), -1)

            avg_x = (rx + lx) / 2.0
            avg_y = (ry + ly) / 2.0

            # Output RAW ratios
            output = {"x": avg_x, "y": avg_y}
            print(json.dumps(output))
            sys.stdout.flush()
            
            if not calibration_mode:
                cv2.putText(final_display, f"Raw X: {avg_x:.4f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(final_display, f"Raw Y: {avg_y:.4f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Iris Tracker", final_display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # Esc
            break
        elif key == 32 or key == 13: # Space or Enter
            # Send trigger to Rust
            print(json.dumps({"type": "trigger"}))
            sys.stdout.flush()


except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
