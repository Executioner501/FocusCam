import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time

# ---------------- LOAD MODEL ---------------- #
MODEL_PATH = "gaze_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

class_names = ["up", "forward", "left", "down", "right"]
IMG_SIZE = (64, 64)

# ---------------- MEDIAPIPE SETUP ---------------- #
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE  = [33, 133]
RIGHT_EYE = [362, 263]

def eye_region_box_static(frame, lm):
    h, w, _ = frame.shape

    lx = int(lm[LEFT_EYE[0]].x * w)
    rx = int(lm[RIGHT_EYE[0]].x * w)
    ly = int(lm[LEFT_EYE[0]].y * h)
    ry = int(lm[RIGHT_EYE[0]].y * h)

    cx = (lx + rx) // 2
    cy = (ly + ry) // 2

    BOX_W = 180
    BOX_H = 90

    x1 = max(cx - BOX_W//2, 0)
    y1 = max(cy - BOX_H//2, 0)
    x2 = min(cx + BOX_W//2, w)
    y2 = min(cy + BOX_H//2, h)

    return x1, y1, x2, y2

# ---------------- DEBOUNCE ---------------- #
last_state = "UNKNOWN"
last_change_time = time.time()
STABLE_DURATION = 0.25   # 250 ms stable before switching state
FORWARD_THRESHOLD = 0.1 # must exceed 60% to count as looking

# ---------------- VIDEO LOOP ---------------- #
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    current_state = "UNKNOWN"

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        
        x1, y1, x2, y2 = eye_region_box_static(frame, lm)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            resized = cv2.resize(crop, IMG_SIZE)
            rgb_crop = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            inp = np.expand_dims(rgb_crop.astype(np.float32), axis=0)

            preds = model(inp, training=False)[0]
            idx = np.argmax(preds)
            gaze_label = class_names[idx]
            conf = preds[idx]

            # -------- Combine MediaPipe + Model -------- #
            if gaze_label == "forward" and conf > FORWARD_THRESHOLD:
                current_state = "LOOKING"
            else:
                current_state = "NOT LOOKING"

    # -------- APPLY DEBOUNCE -------- #
    if current_state != last_state:
        if time.time() - last_change_time > STABLE_DURATION:
            last_state = current_state
        else:
            current_state = last_state
    else:
        last_change_time = time.time()

    # -------- DISPLAY FINAL ANSWER -------- #
    color = (0,255,0) if current_state == "LOOKING" else (0,0,255)
    cv2.putText(frame, current_state, (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Gaze Detector (Combined)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
