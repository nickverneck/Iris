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

def get_head_basis(landmarks, w, h):
    # Rigid points to define the head plane
    # 33: Right Eye Inner (actually MP Right Inner is 133 or 33? 33 is Left-most of right eye i.e. inner)
    # Let's use Outer corners for stability: 33 (Left of Right Eye), 263 (Right of Left Eye) - wait.
    # Standard MP Mesh: 33 = Right Eye Inner, 133 = Right Eye Outer. 362 = Left Eye Inner, 263 = Left Eye Outer.
    # Let's use the two Outer Corners as the horizontal extreme points.
    r_outer = get_landmark_point_3d(landmarks, 133, w, h)
    l_outer = get_landmark_point_3d(landmarks, 263, w, h)
    chin = get_landmark_point_3d(landmarks, 152, w, h)
    
    # 1. Head Right Vector (Left Eye -> Right Eye)
    # Note: MP coords have X increasing to the RIGHT of the image.
    # So L_Outer (on left of screen) has lower X. R_Outer (on right) has higher X.
    # Vector L->R is positive X.
    head_right = l_outer - r_outer # Wait, in mirror mode? 
    # Let's just trust x is x.
    # If 263 is Left Eye (Viewer's Right if selfie), let's just subtract.
    head_right = l_outer - r_outer 
    head_right = head_right / np.linalg.norm(head_right)
    
    # 2. Head Up Vector
    # Midpoint of eyes
    eye_mid = (l_outer + r_outer) / 2.0
    # Vector from Chin to Eye Mid is roughly UP
    head_up_rough = eye_mid - chin
    head_up_rough = head_up_rough / np.linalg.norm(head_up_rough)
    
    # 3. Head Forward (Normal)
    # Cross Right x Up -> Forward (Z)
    head_fwd = np.cross(head_right, head_up_rough)
    head_fwd = head_fwd / np.linalg.norm(head_fwd)
    
    # 4. Re-orthogonalize Up
    # Forward x Right -> True Up
    head_up = np.cross(head_fwd, head_right)
    head_up = head_up / np.linalg.norm(head_up)
    
    return head_right, head_up, head_fwd

def get_relative_iris_pos(eye_center, iris, head_right, head_up, scale_ref):
    # Vector from Eye Center to Iris
    gaze_vec = iris - eye_center
    
    # Project onto Head Axes
    x_score = np.dot(gaze_vec, head_right)
    y_score = np.dot(gaze_vec, head_up)
    
    # Normalize by scale (e.g. eye distance) to be distance-invariant
    x_ratio = x_score / scale_ref
    y_ratio = y_score / scale_ref
    
    # Sensitivity multiplier (iris moves ~10-20% of eye width max)
    # We need to expand this to cover 0.0-1.0 range
    sensitivity = 5.0
    
    # Y-axis inversion: Head "Up" is negative screen Y
    # When looking DOWN (positive y_score), we want higher screen Y
    # So we NEGATE y_ratio
    return 0.5 + (x_ratio * sensitivity), 0.5 - (y_ratio * sensitivity)

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
            # --- Head Basis ---
            b_right, b_up, b_fwd = get_head_basis(landmarks, w, h)
            
            # Scale reference: Distance between outer corners
            # right_outer (133) and left_outer (263)
            # We already computed points in get_head_basis but good to have explicit here or pass it?
            # Let's recompute points for clarity
            pt_r_out = get_landmark_point_3d(landmarks, 133, w, h)
            pt_l_out = get_landmark_point_3d(landmarks, 263, w, h)
            pt_r_in = get_landmark_point_3d(landmarks, 33, w, h)
            pt_l_in = get_landmark_point_3d(landmarks, 362, w, h)
            
            head_scale = np.linalg.norm(pt_l_out - pt_r_out)
            
            # --- Right Eye ---
            r_center = (pt_r_in + pt_r_out) / 2.0
            r_iris = get_landmark_point_3d(landmarks, 468, w, h)
            
            # --- Left Eye ---
            l_center = (pt_l_in + pt_l_out) / 2.0
            l_iris = get_landmark_point_3d(landmarks, 473, w, h)

            rx, ry = get_relative_iris_pos(r_center, r_iris, b_right, b_up, head_scale)
            lx, ly = get_relative_iris_pos(l_center, l_iris, b_right, b_up, head_scale)
            
            # Debug Vis on Camera Frame (if visible)
            if not calibration_mode:
                # Debug Line: Head Up Axis at Center of Face
                face_cx, face_cy = int((r_center[0] + l_center[0])/2), int((r_center[1] + l_center[1])/2)
                up_end = (int(face_cx + b_up[0]*50), int(face_cy + b_up[1]*50))
                cv2.arrowedLine(frame, (face_cx, face_cy), up_end, (255, 0, 0), 2) # Blue Up
                
                # Debug Line: Gaze Vector
                r_i_2d = (int(r_iris[0]), int(r_iris[1]))
                l_i_2d = (int(l_iris[0]), int(l_iris[1]))
                cv2.circle(frame, r_i_2d, 2, (0, 255, 0), -1)
                cv2.circle(frame, l_i_2d, 2, (0, 255, 0), -1)

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
