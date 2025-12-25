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
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
# Start in full screen for calibration visibility
cv2.namedWindow("Iris Tracker", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Iris Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

calibration_mode = False
calib_dot = None # (x, y) relative 0-1

def get_landmark_point_3d(landmarks, idx, w, h):
    point = landmarks[idx]
    # MP coords: x,y are [0,1], z is "roughly same scale as x"
    return np.array([point.x * w, point.y * h, point.z * w])

def get_relative_iris_pos(inner, outer, iris):
    # 1. Center of the eye (surface approximation)
    eye_center = (inner + outer) / 2.0
    
    # 2. Eye Width Vector (Inner -> Outer) - This defines the "Horizontal" of the eye
    # We use this to correct for Head Yaw/Roll
    eye_width_vec = outer - inner
    
    # 3. Iris Vector (Center -> Iris)
    iris_vec = iris - eye_center
    
    # 4. Project Iris movement onto the Eye's coordinate system
    
    # Basis X: Normalized Eye Width Vector
    basis_x = eye_width_vec / np.linalg.norm(eye_width_vec)
    
    # Basis Y: orthogonal to X (in the general "up" direction)
    # We need a temporary Up vector. (0, -1, 0) in screen space is up.
    # But head might be tilted.
    # Let's approximate "Up" as perpendicular to Basis X in the Z-plane?
    # Better: Use the cross product with a rough Forward vector?
    # Simple approach: Just project onto X, and then use the residual for Y.
    
    # Project iris_vec onto basis_x
    x_proj = np.dot(iris_vec, basis_x)
    
    # The "Vertical" component is the perpendicular part?
    # Not exactly, because the eye is a sphere.
    # Let's convert to normalized coordinates.
    
    # Normalize by eye width (scale invariant)
    eye_width = np.linalg.norm(eye_width_vec)
    
    # X Ratio: How far along the width vector?
    # (Centered at 0)
    x_score = x_proj / eye_width 
    
    # Y Score: The vertical displacement
    # Remove the X component from the vector
    y_vec = iris_vec - (basis_x * x_proj)
    
    # Ideally we'd project this onto a true "Eye Up" vector, but we don't have one easily.
    # We can assume "Eye Up" is roughly (0, 1, 0) locally?
    # Let's just take the Y-component of the residual vector, 
    # BUT we must account for head roll.
    # Alternative: Cross product of Basis X and Z-axis?
    # Let's try simple projection onto the world Y axis first, corrected by scale.
    # Actually, simplistic: Just taking component perpendicular to eye-width is good enough for now.
    # Sign? In screen space Y is down.
    # If y_vec[1] is positive -> down.
    
    # Let's use the magnitude of the residual, signed by its Y component.
    y_mag = np.linalg.norm(y_vec)
    y_sign = np.sign(y_vec[1]) 
    y_score = (y_mag * y_sign) / eye_width

    # Shift to 0.5 center for compatibility with existing calibration logic
    # Previous range was 0.0 to 1.0. 
    # Now it's -0.5 to 0.5 approx.
    return 0.5 + x_score, 0.5 + y_score

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
                
                # Draw large target
                cv2.circle(canvas, (cx, cy), 50, (0, 0, 255), -1)      # Red body
                cv2.circle(canvas, (cx, cy), 15, (255, 255, 255), -1)  # White center
                cv2.circle(canvas, (cx, cy), 55, (255, 255, 255), 2)   # White ring
                
                text = "Look at DOT & press SPACE"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = cx - text_size[0] // 2
                text_y = cy + 100
                
                cv2.putText(canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            final_display = canvas
        else:
            final_display = frame

        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # ... (Landmark extraction code same as before) ...
            # --- Right Eye ---
            right_inner = get_landmark_point_3d(landmarks, 33, w, h)
            right_outer = get_landmark_point_3d(landmarks, 133, w, h)
            right_iris = get_landmark_point_3d(landmarks, 468, w, h)
            
            # --- Left Eye ---
            left_inner = get_landmark_point_3d(landmarks, 362, w, h)
            left_outer = get_landmark_point_3d(landmarks, 263, w, h)
            left_iris = get_landmark_point_3d(landmarks, 473, w, h)

            rx, ry = get_relative_iris_pos(right_inner, right_outer, right_iris)
            lx, ly = get_relative_iris_pos(left_inner, left_outer, left_iris)
            
            # Debug Vis on Camera Frame (if visible)
            if not calibration_mode:
                # Project back to 2D for drawing
                r_center = (right_inner[:2] + right_outer[:2]) / 2.0
                l_center = (left_inner[:2] + left_outer[:2]) / 2.0
                r_iris_2d = right_iris[:2]
                l_iris_2d = left_iris[:2]

                # Draw Eye Center to Iris vector
                cv2.arrowedLine(frame, (int(r_center[0]), int(r_center[1])), (int(r_iris_2d[0]), int(r_iris_2d[1])), (0, 255, 255), 2)
                cv2.arrowedLine(frame, (int(l_center[0]), int(l_center[1])), (int(l_iris_2d[0]), int(l_iris_2d[1])), (0, 255, 255), 2)
                
                # Draw Iris points
                cv2.circle(frame, (int(r_iris_2d[0]), int(r_iris_2d[1])), 2, (0, 255, 0), -1)
                cv2.circle(frame, (int(l_iris_2d[0]), int(l_iris_2d[1])), 2, (0, 255, 0), -1)

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
