import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# --- 1. Load Your Trained Model ---
MODEL_PATH = 'yawn_model.h5' # Make sure this file is in the same folder
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 2. Setup MediaPipe & Webcam ---
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- 3. Define Constants ---
CROP_SIZE = 64  # The same size your model was trained on
IMG_SIZE = (CROP_SIZE, CROP_SIZE)
FONT = cv2.FONT_HERSHEY_SIMPLEX
THRESHOLD = 0.2 # Confidence threshold for 'yawn'
# Note: Check your 'Classes found:' output. 
# If 'yawn' is 0 and 'no_yawn' is 1, change this to:
# if confidence < THRESHOLD: 

# --- 4. Main Loop ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1) # Flip for mirror view
    image_height, image_width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # For MediaPipe
    results = face_detection.process(image_rgb)

    prediction_text = "Not Yawning"
    text_color = (0, 255, 0) # Green

    if results.detections:
        for detection in results.detections:

            bbox = detection.location_data.relative_bounding_box

            x_min = int(bbox.xmin * image_width)
            y_min = int(bbox.ymin * image_height)
            w = int(bbox.width * image_width)
            h = int(bbox.height * image_height)

            # --- Mouth region inside face bounding box ---
            mx1 = x_min + int(0.25 * w)
            mx2 = x_min + int(0.75 * w)
            my1 = y_min + int(0.60 * h)
            my2 = y_min + int(0.95 * h)

            # Safety check
            if mx1 < 0 or my1 < 0 or mx2 > image_width or my2 > image_height:
                continue

            # 1. Crop mouth region
            mouth_crop = image[my1:my2, mx1:mx2]

            # 2. Resize to model size
            mouth_crop_resized = cv2.resize(mouth_crop, IMG_SIZE)

            # 3. Convert BGR → RGB
            mouth_crop_rgb = cv2.cvtColor(mouth_crop_resized, cv2.COLOR_BGR2RGB)

            # 4. Convert to float32
            mouth_crop_float = mouth_crop_rgb.astype(np.float32)

            # 5. Create batch dimension
            input_tensor = np.expand_dims(mouth_crop_float, axis=0)

            # --- Prediction ---
            prediction = model.predict(input_tensor)
            confidence = prediction[0][0]

            # --- Interpret ---
            if confidence > THRESHOLD:
                prediction_text = f"YAWNING ({confidence*100:.0f}%)"
                text_color = (0, 0, 255)
            else:
                prediction_text = f"Not Yawning ({(1-confidence)*100:.0f}%)"
                text_color = (0, 255, 0)

            # draw NEW mouth box
            cv2.rectangle(image, (mx1, my1), (mx2, my2), text_color, 2)

            break


    # --- 8. Display the Result ---
    cv2.putText(image, prediction_text, (10, 30), FONT, 1, text_color, 2, cv2.LINE_AA)
    cv2.imshow('Yawn Detector Test', image)

    if cv2.waitKey(5) & 0xFF == 27: # Press 'ESC' to quit
        break

# --- 9. Cleanup ---
cap.release()
cv2.destroyAllWindows()