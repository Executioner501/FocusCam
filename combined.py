# combined_newgaze.py
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
from pathlib import Path

# Try import YOLO (ultralytics). If missing, phone detection will be disabled.
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# ---------------- PATHS / CONFIG -----------------
GAZE_MODEL_PATH = "gaze_model_1.keras"   # your gaze model file (Keras)
YAWN_MODEL_PATH = "yawn_model.h5"      # your yawn detector file (H5)
YOLO_MODEL_PATH  = "yolov8n.pt"        # yolov8 model file (or rely on automatic download)

IMG_SIZE = (64, 64)                    # gaze DL model input (w,h)
STABLE_DURATION = 0.18                 # debounce seconds
OPENCV_WEIGHT = 0.70                   # weight for iris-based decision
DL_WEIGHT = 0.30                       # weight for gaze DL
FORWARD_LABEL = "forward"              # label string from your gaze DL

# ---------------- LOAD MODELS -------------------
# Load Gaze DL model (optional)
MODEL_AVAILABLE = False
class_names = None
try:
    gaze_model = tf.keras.models.load_model(GAZE_MODEL_PATH)
    MODEL_AVAILABLE = True
    # ensure class order matches training: user said ["up","forward","left","down","right"]
    class_names = ["up", "forward", "left", "down", "right"]
    print("Loaded gaze DL model:", GAZE_MODEL_PATH)
except Exception as e:
    print("Gaze model not loaded:", e)

# Load yawn model (optional)
yawn_available = False
try:
    yawn_model = tf.keras.models.load_model(YAWN_MODEL_PATH)
    yawn_available = True
    print("Loaded yawn model:", YAWN_MODEL_PATH)
except Exception as e:
    print("Yawn model not loaded:", e)

# Load YOLO for phone detection (optional)
phone_available = False
if YOLO_AVAILABLE:
    try:
        phone_model = YOLO(YOLO_MODEL_PATH)  # will download if needed
        phone_available = True
        print("Loaded YOLO phone model.")
    except Exception as e:
        print("YOLO not loaded:", e)
else:
    print("ultralytics.YOLO not available; phone detection disabled.")

# ---------------- MEDIAPIPE SETUP -----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,        # iris landmarks available
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_face_detection = mp.solutions.face_detection  # used for mouth box

# Landmark indices (MediaPipe)
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT_EYE_CORNERS  = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]
LEFT_EYE_VERT = [159, 145]
RIGHT_EYE_VERT = [386, 374]

# Gaze box / target zone parameters (tweak to be stricter/looser)
H_MIN, H_MAX = 0.38, 0.62
V_MIN, V_MAX = 0.40, 0.70

# ---------------- HELPER FUNCTIONS ----------------
def get_gaze_ratio(iris_center, left_corner, right_corner, top_point, bottom_point):
    """Return (h_ratio, v_ratio) clipped in [0,1]."""
    eye_width = max(right_corner[0] - left_corner[0], 1)
    eye_height = max(bottom_point[1] - top_point[1], 1)
    h_ratio = (iris_center[0] - left_corner[0]) / eye_width
    v_ratio = (iris_center[1] - top_point[1]) / eye_height
    return float(np.clip(h_ratio, 0.0, 1.0)), float(np.clip(v_ratio, 0.0, 1.0))

def classify_gaze_direction(h_ratio, v_ratio):
    """Return direction label and confidence (forward/left/right/up/down)."""
    h_centered = H_MIN < h_ratio < H_MAX
    v_centered = V_MIN < v_ratio < V_MAX
    if h_centered and v_centered:
        # more centered -> higher confidence
        h_distance = abs(h_ratio - 0.5)
        v_distance = abs(v_ratio - 0.55)
        confidence = 0.95 - (h_distance + v_distance) * 0.5
        return "forward", max(0.70, confidence)
    violations = []
    if v_ratio < V_MIN: violations.append(("up", V_MIN - v_ratio))
    if v_ratio > V_MAX: violations.append(("down", v_ratio - V_MAX))
    if h_ratio < H_MIN: violations.append(("right", H_MIN - h_ratio))
    if h_ratio > H_MAX: violations.append(("left", h_ratio - H_MAX))
    if violations:
        dir_name = max(violations, key=lambda x: x[1])[0]
        return dir_name, 0.80
    return "forward", 0.50

def determine_gaze_state(left_h, left_v, right_h, right_v):
    """Return state, color, confidence, reason, direction."""
    if left_h is None and right_h is None:
        return "UNKNOWN", (128,128,128), 0.0, "No eyes", "unknown"
    if left_h is not None and right_h is not None:
        avg_h = (left_h + right_h) / 2.0
        avg_v = (left_v + right_v) / 2.0
        direction, conf = classify_gaze_direction(avg_h, avg_v)
        if direction == "forward":
            return "LOOKING", (0,255,0), conf, f"Forward H:{avg_h:.2f} V:{avg_v:.2f}", direction
        else:
            return "NOT_LOOKING", (0,0,255), conf, f"Looking {direction.upper()} H:{avg_h:.2f} V:{avg_v:.2f}", direction
    # single-eye fallback
    h = left_h if left_h is not None else right_h
    v = left_v if left_v is not None else right_v
    direction, conf = classify_gaze_direction(h, v)
    if direction == "forward":
        return "LOOKING", (0,255,0), conf * 0.8, "Single eye forward", direction
    else:
        return "NOT_LOOKING", (0,0,255), conf * 0.8, f"Single eye {direction}", direction

def get_dl_prediction(frame, lm):
    """Return (label, confidence) from gaze DL model if available."""
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
    x1 = max(cx - BOX_W//2, 0); y1 = max(cy - BOX_H//2, 0)
    x2 = min(cx + BOX_W//2, w); y2 = min(cy + BOX_H//2, h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, 0.0
    resized = cv2.resize(crop, (IMG_SIZE[0], IMG_SIZE[1]))
    rgb_crop = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    inp = np.expand_dims(rgb_crop.astype(np.float32), axis=0)
    preds = gaze_model(inp, training=False)[0]
    idx = int(np.argmax(preds))
    return class_names[idx], float(preds[idx])

def combine_predictions(cv_state, cv_conf, cv_direction, dl_label, dl_conf):
    """Weighted combination: OpenCV (iris) + DL (gaze model)."""
    if dl_label is None or cv_state == "UNKNOWN":
        return cv_state, cv_conf
    cv_looking = (cv_direction == "forward")
    dl_looking = (dl_label == FORWARD_LABEL)
    cv_score = cv_conf if cv_looking else (1 - cv_conf)
    dl_score = dl_conf if dl_looking else (1 - dl_conf)
    combined = (OPENCV_WEIGHT * cv_score) + (DL_WEIGHT * dl_score)
    final_state = "LOOKING" if combined > 0.5 else "NOT_LOOKING"
    return final_state, float(combined)

def draw_target_zone(frame):
    """Draw a semi-transparent target zone for the user to look at."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    center_x = w // 2; center_y = h // 2
    zone_w, zone_h = 120, 80
    cv2.rectangle(overlay, (center_x-zone_w, center_y-zone_h), (center_x+zone_w, center_y+zone_h), (0,255,0), -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.rectangle(frame, (center_x-zone_w, center_y-zone_h), (center_x+zone_w, center_y+zone_h), (0,255,0), 2)
    cv2.putText(frame, "Look at GREEN ZONE = LOOKING", (center_x-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

# ---------------- MAIN LOOP ----------------
cap = cv2.VideoCapture(0)
last_state = "UNKNOWN"
last_change_time = time.time()

print("=== Combined attention monitor running ===")
print("Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # defaults
    looking_status = "UNKNOWN"; yawn_status = "OK"; phone_status = "OK"
    cv_direction = "unknown"; cv_conf = 0.0

    # A) GAZE (MediaPipe iris)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        # left iris, corners, vertical top/bottom
        lix = int(lm[LEFT_IRIS[0]].x * w); liy = int(lm[LEFT_IRIS[0]].y * h)
        lcl = (int(lm[LEFT_EYE_CORNERS[0]].x * w), int(lm[LEFT_EYE_CORNERS[0]].y * h))
        lcr = (int(lm[LEFT_EYE_CORNERS[1]].x * w), int(lm[LEFT_EYE_CORNERS[1]].y * h))
        ltop = (int(lm[LEFT_EYE_VERT[0]].x * w), int(lm[LEFT_EYE_VERT[0]].y * h))
        lbot = (int(lm[LEFT_EYE_VERT[1]].x * w), int(lm[LEFT_EYE_VERT[1]].y * h))

        # right iris
        rix = int(lm[RIGHT_IRIS[0]].x * w); riy = int(lm[RIGHT_IRIS[0]].y * h)
        rcl = (int(lm[RIGHT_EYE_CORNERS[0]].x * w), int(lm[RIGHT_EYE_CORNERS[0]].y * h))
        rcr = (int(lm[RIGHT_EYE_CORNERS[1]].x * w), int(lm[RIGHT_EYE_CORNERS[1]].y * h))
        rtop = (int(lm[RIGHT_EYE_VERT[0]].x * w), int(lm[RIGHT_EYE_VERT[0]].y * h))
        rbot = (int(lm[RIGHT_EYE_VERT[1]].x * w), int(lm[RIGHT_EYE_VERT[1]].y * h))

        left_h, left_v = get_gaze_ratio((lix, liy), lcl, lcr, ltop, lbot)
        right_h, right_v = get_gaze_ratio((rix, riy), rcl, rcr, rtop, rbot)

        # visualize iris centers
        cv2.circle(frame, (lix, liy), 3, (255,0,255), -1)
        cv2.circle(frame, (rix, riy), 3, (255,0,255), -1)

        looking_status, color, cv_conf, reason, cv_direction = determine_gaze_state(left_h, left_v, right_h, right_v)

        # get DL prediction (low-weight contribution)
        dl_label, dl_conf = get_dl_prediction(frame, lm) if MODEL_AVAILABLE else (None, 0.0)
    else:
        draw_target_zone(frame)
        dl_label, dl_conf = (None, 0.0)

    # B) YAWN detection (MediaPipe face detection -> crop mouth -> yawn_model)
    if yawn_available:
        fd = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.45)
        res_fd = fd.process(rgb)
        if res_fd.detections:
            for det in res_fd.detections:
                box = det.location_data.relative_bounding_box
                bx = int(box.xmin * w); by = int(box.ymin * h)
                bw = int(box.width * w); bh = int(box.height * h)
                mx1 = max(0, bx + int(0.25 * bw)); mx2 = min(w, bx + int(0.75 * bw))
                my1 = max(0, by + int(0.60 * bh)); my2 = min(h, by + int(0.95 * bh))
                crop = frame[my1:my2, mx1:mx2]
                if crop.size != 0:
                    c = cv2.resize(crop, IMG_SIZE)
                    c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
                    inp = np.expand_dims(c.astype(np.float32), axis=0)
                    pred = float(yawn_model(inp, training=False).numpy()[0][0])
                    yawn_status = "YAWNING" if pred > 0.33 else "OK"
                break

    # C) Phone detection (YOLO)
    if phone_available:
        try:
            results_phone = phone_model(frame, stream=True, verbose=False)
            # iterate detections
            for r in results_phone:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = phone_model.names[cls_id]
                    if cls_name == "cell phone":
                        phone_status = "PHONE"
                        break
                if phone_status == "PHONE":
                    break
        except Exception:
            phone_status = "OK"

    # FINAL combination and decision
    final_state = "UNKNOWN"; final_color = (128,128,128)
    combined_conf = cv_conf
    if MODEL_AVAILABLE and results.multi_face_landmarks:
        # combine OpenCV + DL
        final_state, combined_conf = combine_predictions(looking_status, cv_conf, cv_direction, dl_label, dl_conf)
    else:
        final_state = looking_status
        combined_conf = cv_conf

    # prioritize phone > yawn > gaze
    if phone_status == "PHONE":
        display_text = "DISTRACTED: PHONE"
        display_color = (0,0,255)
    elif yawn_status == "YAWNING":
        display_text = "DISTRACTED: YAWNING"
        display_color = (0,165,255)
    elif final_state == "NOT_LOOKING":
        display_text = "DISTRACTED: LOOKING AWAY"
        display_color = (0,0,255)
    elif final_state == "LOOKING":
        display_text = "PAYING ATTENTION"
        display_color = (0,255,0)
    else:
        display_text = "UNKNOWN"
        display_color = (128,128,128)

    # Debounce: avoid flicker
    if display_text != last_state and display_text != "UNKNOWN":
        if time.time() - last_change_time > STABLE_DURATION:
            last_state = display_text
            last_change_time = time.time()
    else:
        last_change_time = time.time()

    # DRAW HUD / STATS
    cv2.putText(frame, f"{display_text}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, display_color, 3)
    # show simple diagnostics
    y0 = 70
    cv2.putText(frame, f"Gaze DL: {dl_label if dl_label else 'N/A'} {dl_conf:.2f}", (10,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1); y0 += 20
    cv2.putText(frame, f"Iris state: {looking_status} ({cv_conf:.2f})", (10,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1); y0 += 20
    cv2.putText(frame, f"Yawn: {yawn_status}", (10,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1); y0 += 20
    cv2.putText(frame, f"Phone: {phone_status}", (10,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    cv2.imshow("Combined Attention Monitor", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
