import cv2
import mediapipe as mp
import os
import time

# -------- SETTINGS --------
SAVE_DIR = "eye_dataset"       # root folder where images are saved
IMG_SIZE = (64, 64)            # size of cropped eye images
FPS_THROTTLE = 5               # save every 5 frames after key press
# --------------------------

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # gives iris landmarks!
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Controls:")
print("[f] Looking Forward")
print("[l] Looking Left")
print("[r] Looking Right")
print("[u] Looking Up")
print("[d] Looking Down")
print("[q] Quit")

current_label = None
frame_counter = 0

# Utility: create label folders automatically
def ensure_label_folders():
    labels = ["forward", "left", "right", "up", "down"]
    for lab in labels:
        path = os.path.join(SAVE_DIR, lab)
        os.makedirs(path, exist_ok=True)

ensure_label_folders()

# Eye landmark indices
# From MediaPipe FaceMesh reference
LEFT_EYE = [33, 133, 160, 159, 158, 157, 173]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398]

def crop_eye(frame, landmarks, eye_points):
    h, w, _ = frame.shape
    xs, ys = [], []

    for idx in eye_points:
        lm = landmarks[idx]
        xs.append(int(lm.x * w))
        ys.append(int(lm.y * h))

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    # Padding around eye box
    pad = 10
    x1 -= pad; y1 -= pad
    x2 += pad; y2 += pad

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    eye_crop = frame[y1:y2, x1:x2]
    eye_crop = cv2.resize(eye_crop, IMG_SIZE)

    return eye_crop


while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0].landmark

        # Crop both eyes
        left_eye_img = crop_eye(frame, mesh, LEFT_EYE)
        right_eye_img = crop_eye(frame, mesh, RIGHT_EYE)

        # Show preview windows
        cv2.imshow("Left Eye", left_eye_img)
        cv2.imshow("Right Eye", right_eye_img)

        # Save eyes if a label is active
        if current_label is not None:
            frame_counter += 1
            if frame_counter % FPS_THROTTLE == 0:
                timestamp = int(time.time() * 1000)
                cv2.imwrite(f"{SAVE_DIR}/{current_label}/left_{timestamp}.jpg", left_eye_img)
                cv2.imwrite(f"{SAVE_DIR}/{current_label}/right_{timestamp}.jpg", right_eye_img)
                print(f"Saved: {current_label}")

    # Show main frame
    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1) & 0xFF

    # Change labels
    if key == ord('f'):
        current_label = "forward"
        print("Label = forward")
    elif key == ord('l'):
        current_label = "left"
        print("Label = left")
    elif key == ord('r'):
        current_label = "right"
        print("Label = right")
    elif key == ord('u'):
        current_label = "up"
        print("Label = up")
    elif key == ord('d'):
        current_label = "down"
        print("Label = down")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
