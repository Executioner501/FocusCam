import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time

# ---------------- LOAD MODEL (LOW CONTRIBUTION) ---------------- #
try:
    MODEL_PATH = "gaze_model_1.keras"
    model = tf.keras.models.load_model(MODEL_PATH)
    print("DL Model loaded (10% contribution)")
    class_names = ["up", "forward", "left", "down", "right"]
    IMG_SIZE = (64, 64)
    MODEL_AVAILABLE = True
except:
    print("DL Model not found, using pure OpenCV")
    MODEL_AVAILABLE = False

# ---------------- MEDIAPIPE SETUP ---------------- #
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Iris landmarks
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Eye landmarks
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]
LEFT_EYE_VERTICAL = [159, 145]
RIGHT_EYE_VERTICAL = [386, 374]

# ========== GAZE BOUNDARIES ========== #
# These define the "LOOKING AT SCREEN" zone
H_MIN = 0.38  # Looser horizontal (was 0.42)
H_MAX = 0.62  # Looser horizontal (was 0.58)
V_MIN = 0.40  # Looser vertical (was 0.45)
V_MAX = 0.70  # Looser vertical (was 0.65)

def get_gaze_ratio(iris_center, left_corner, right_corner, top_point, bottom_point):
    """Calculate 2D gaze position"""
    # Horizontal
    iris_to_left = iris_center[0] - left_corner[0]
    eye_width = right_corner[0] - left_corner[0]
    h_ratio = (iris_to_left / eye_width) if eye_width != 0 else 0.5
    h_ratio = max(0.0, min(1.0, h_ratio))
    
    # Vertical
    iris_to_top = iris_center[1] - top_point[1]
    eye_height = bottom_point[1] - top_point[1]
    v_ratio = (iris_to_top / eye_height) if eye_height != 0 else 0.5
    v_ratio = max(0.0, min(1.0, v_ratio))
    
    return h_ratio, v_ratio

def classify_gaze_direction(h_ratio, v_ratio):
    """
    Classify gaze into 5 directions: forward, left, right, up, down
    More lenient boundaries with tolerance zones
    """
    h_centered = H_MIN < h_ratio < H_MAX
    v_centered = V_MIN < v_ratio < V_MAX
    
    # FORWARD = both centered
    if h_centered and v_centered:
        # More centered = higher confidence
        h_distance = abs(h_ratio - 0.5)
        v_distance = abs(v_ratio - 0.55)
        confidence = 0.95 - (h_distance + v_distance) * 0.5
        return "forward", max(0.75, confidence)
    
    # Calculate distances from boundaries to prioritize
    h_diff_left = h_ratio - H_MIN
    h_diff_right = H_MAX - h_ratio
    v_diff_top = v_ratio - V_MIN
    v_diff_bottom = V_MAX - v_ratio
    
    # Find which boundary is violated most
    violations = []
    if v_ratio < V_MIN:
        violations.append(("up", V_MIN - v_ratio))
    if v_ratio > V_MAX:
        violations.append(("down", v_ratio - V_MAX))
    if h_ratio < H_MIN:
        violations.append(("right", H_MIN - h_ratio))
    if h_ratio > H_MAX:
        violations.append(("left", h_ratio - H_MAX))
    
    if violations:
        # Return the most violated direction
        direction = max(violations, key=lambda x: x[1])[0]
        return direction, 0.80
    
    # Should not reach here
    return "forward", 0.50

def determine_gaze_state(left_h, left_v, right_h, right_v):
    """Determine final gaze state using both eyes"""
    if left_h is None and right_h is None:
        return "UNKNOWN", (128, 128, 128), 0.0, "No eyes", "unknown"
    
    # Use both eyes if available
    if left_h is not None and right_h is not None:
        # Average the positions
        avg_h = (left_h + right_h) / 2
        avg_v = (left_v + right_v) / 2
        
        direction, conf = classify_gaze_direction(avg_h, avg_v)
        
        if direction == "forward":
            return "LOOKING", (0, 255, 0), conf, f"Forward H:{avg_h:.2f} V:{avg_v:.2f}", direction
        else:
            return "NOT LOOKING", (0, 0, 255), conf, f"Looking {direction.upper()} H:{avg_h:.2f} V:{avg_v:.2f}", direction
    
    # Single eye
    h = left_h if left_h is not None else right_h
    v = left_v if left_v is not None else right_v
    eye_name = "L" if left_h is not None else "R"
    
    direction, conf = classify_gaze_direction(h, v)
    
    if direction == "forward":
        return "LOOKING", (0, 255, 0), conf * 0.8, f"Single {eye_name} forward", direction
    else:
        return "NOT LOOKING", (0, 0, 255), conf * 0.8, f"Single {eye_name} {direction}", direction

def get_dl_prediction(frame, lm):
    """Get DL model prediction"""
    if not MODEL_AVAILABLE:
        return None, 0.0
    
    h, w, _ = frame.shape
    lx = int(lm[LEFT_EYE_CORNERS[0]].x * w)
    rx = int(lm[RIGHT_EYE_CORNERS[0]].x * w)
    ly = int(lm[LEFT_EYE_CORNERS[0]].y * h)
    ry = int(lm[RIGHT_EYE_CORNERS[0]].y * h)

    cx = (lx + rx) // 2
    cy = (ly + ry) // 2
    BOX_W, BOX_H = 180, 90

    x1 = max(cx - BOX_W//2, 0)
    y1 = max(cy - BOX_H//2, 0)
    x2 = min(cx + BOX_W//2, w)
    y2 = min(cy + BOX_H//2, h)

    crop = frame[y1:y2, x1:x2]
    if crop.size > 0:
        resized = cv2.resize(crop, IMG_SIZE)
        rgb_crop = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(rgb_crop.astype(np.float32), axis=0)
        preds = model(inp, training=False)[0]
        idx = np.argmax(preds)
        return class_names[idx], float(preds[idx])
    
    return None, 0.0

def combine_predictions(cv_state, cv_conf, cv_direction, dl_label, dl_conf):
    """Combine OpenCV (80%) with DL (20%) - balanced approach"""
    if dl_label is None or cv_state == "UNKNOWN":
        return cv_state, cv_conf
    
    # Check if both agree on direction
    cv_looking = (cv_direction == "forward")
    dl_looking = (dl_label == "forward")
    
    # More balanced weights
    OPENCV_WEIGHT = 0.70
    DL_WEIGHT = 0.30
    
    cv_score = cv_conf if cv_looking else (1 - cv_conf)
    dl_score = dl_conf if dl_looking else (1 - dl_conf)
    
    combined = (OPENCV_WEIGHT * cv_score) + (DL_WEIGHT * dl_score)
    
    # Lower threshold for "LOOKING"
    final_state = "LOOKING" if combined > 0.50 else "NOT LOOKING"
    return final_state, combined

def draw_target_zone(frame, h, w):
    """Draw visual target zone on screen showing where to look"""
    # Create semi-transparent overlay
    overlay = frame.copy()
    
    # Calculate center zone (where you should be looking)
    center_x = w // 2
    center_y = h // 2
    
    # Draw GREEN zone (LOOKING zone)
    zone_w = 120
    zone_h = 80
    cv2.rectangle(overlay, 
                  (center_x - zone_w, center_y - zone_h),
                  (center_x + zone_w, center_y + zone_h),
                  (0, 255, 0), -1)
    
    # Blend with original
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    
    # Draw border
    cv2.rectangle(frame,
                  (center_x - zone_w, center_y - zone_h),
                  (center_x + zone_w, center_y + zone_h),
                  (0, 255, 0), 2)
    
    # Add directional labels
    cv2.putText(frame, "UP", (center_x - 20, center_y - zone_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, "DOWN", (center_x - 30, center_y + zone_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, "LEFT", (center_x - zone_w - 60, center_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, "RIGHT", (center_x + zone_w + 10, center_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Instructions
    cv2.putText(frame, "Look at GREEN ZONE = LOOKING", (center_x - 180, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# ---------------- MAIN LOOP ---------------- #
cap = cv2.VideoCapture(0)

last_state = "UNKNOWN"
last_change_time = time.time()
STABLE_DURATION = 0.15  # Faster response (was 0.2)

print("\n=== Balanced Gaze Detector ===")
print(f"LOOKING zone: H=[{H_MIN:.2f}, {H_MAX:.2f}], V=[{V_MIN:.2f}, {V_MAX:.2f}]")
print("Looser boundaries for better detection")
print("Watch for GREEN zone on screen - look at it to trigger LOOKING")
print("Press ESC to exit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    current_state = "UNKNOWN"
    cv_direction = "unknown"

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        
        # Get iris positions
        left_iris_x = int(lm[LEFT_IRIS[0]].x * w)
        left_iris_y = int(lm[LEFT_IRIS[0]].y * h)
        right_iris_x = int(lm[RIGHT_IRIS[0]].x * w)
        right_iris_y = int(lm[RIGHT_IRIS[0]].y * h)
        
        # Get eye landmarks
        left_corner_left = (int(lm[LEFT_EYE_CORNERS[0]].x * w), int(lm[LEFT_EYE_CORNERS[0]].y * h))
        left_corner_right = (int(lm[LEFT_EYE_CORNERS[1]].x * w), int(lm[LEFT_EYE_CORNERS[1]].y * h))
        left_top = (int(lm[LEFT_EYE_VERTICAL[0]].x * w), int(lm[LEFT_EYE_VERTICAL[0]].y * h))
        left_bottom = (int(lm[LEFT_EYE_VERTICAL[1]].x * w), int(lm[LEFT_EYE_VERTICAL[1]].y * h))
        
        right_corner_left = (int(lm[RIGHT_EYE_CORNERS[0]].x * w), int(lm[RIGHT_EYE_CORNERS[0]].y * h))
        right_corner_right = (int(lm[RIGHT_EYE_CORNERS[1]].x * w), int(lm[RIGHT_EYE_CORNERS[1]].y * h))
        right_top = (int(lm[RIGHT_EYE_VERTICAL[0]].x * w), int(lm[RIGHT_EYE_VERTICAL[0]].y * h))
        right_bottom = (int(lm[RIGHT_EYE_VERTICAL[1]].x * w), int(lm[RIGHT_EYE_VERTICAL[1]].y * h))
        
        # Calculate ratios
        left_h, left_v = get_gaze_ratio((left_iris_x, left_iris_y), 
                                        left_corner_left, left_corner_right, 
                                        left_top, left_bottom)
        right_h, right_v = get_gaze_ratio((right_iris_x, right_iris_y),
                                          right_corner_left, right_corner_right,
                                          right_top, right_bottom)
        
        # Visualize
        cv2.circle(frame, (left_iris_x, left_iris_y), 4, (255, 0, 255), -1)
        cv2.circle(frame, (right_iris_x, right_iris_y), 4, (255, 0, 255), -1)
        
        # Determine gaze
        cv_state, color, cv_conf, reason, cv_direction = determine_gaze_state(
            left_h, left_v, right_h, right_v)
        
        # Get DL prediction
        dl_label, dl_conf = get_dl_prediction(frame, lm)
        
        # Combine
        current_state, combined_conf = combine_predictions(
            cv_state, cv_conf, cv_direction, dl_label, dl_conf)
        
        # Draw target zone
        draw_target_zone(frame, h, w)
        
        # Display debug
        y_pos = h - 150
        
        # Show exact values with color coding
        left_h_color = (0, 255, 0) if H_MIN < left_h < H_MAX else (0, 0, 255)
        left_v_color = (0, 255, 0) if V_MIN < left_v < V_MAX else (0, 0, 255)
        right_h_color = (0, 255, 0) if H_MIN < right_h < H_MAX else (0, 0, 255)
        right_v_color = (0, 255, 0) if V_MIN < right_v < V_MAX else (0, 0, 255)
        
        cv2.putText(frame, f"Left Eye:", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"H:{left_h:.3f}", (100, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_h_color, 2)
        cv2.putText(frame, f"V:{left_v:.3f}", (220, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_v_color, 2)
        y_pos += 25
        
        cv2.putText(frame, f"Right Eye:", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"H:{right_h:.3f}", (100, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_h_color, 2)
        cv2.putText(frame, f"V:{right_v:.3f}", (220, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_v_color, 2)
        y_pos += 30
        
        cv2.putText(frame, f"Direction: {cv_direction.upper()}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_pos += 25
        
        if dl_label:
            cv2.putText(frame, f"DL: {dl_label} ({dl_conf:.2f})", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    else:
        draw_target_zone(frame, h, w)

    # Debounce
    if current_state != last_state and current_state != "UNKNOWN":
        if time.time() - last_change_time > STABLE_DURATION:
            last_state = current_state
            last_change_time = time.time()
    else:
        last_change_time = time.time()

    # Display final state - LARGE
    display_color = (0, 255, 0) if last_state == "LOOKING" else (0, 0, 255)
    if last_state == "UNKNOWN":
        display_color = (128, 128, 128)
    
    cv2.putText(frame, last_state, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, display_color, 4)

    cv2.imshow("Balanced Gaze Detector", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()