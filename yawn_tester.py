import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# --- 1. Load Your Trained Model ---
MODEL_PATH = 'my_yawn_model.h5' # Make sure this file is in the same folder
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
THRESHOLD = 0.25 # Confidence threshold for 'yawn'
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
            mouth_center_norm = mp_face_detection.get_key_point(
                detection, mp_face_detection.FaceKeyPoint.MOUTH_CENTER)
            
            if mouth_center_norm:
                mouth_x = int(mouth_center_norm.x * image_width)
                mouth_y = int(mouth_center_norm.y * image_height)
                
                half_size = CROP_SIZE // 2
                y1, y2 = mouth_y - half_size, mouth_y + half_size
                x1, x2 = mouth_x - half_size, mouth_x + half_size
                
                if y1 < 0 or y2 > image_height or x1 < 0 or x2 > image_width:
                    continue 

                # --- 5. Crop and Preprocess (ALL FIXES APPLIED) ---
                
                # 1. Crop
                mouth_crop = image[y1:y2, x1:x2]
                
                # 2. Resize
                mouth_crop_resized = cv2.resize(mouth_crop, IMG_SIZE)
                
                # 3. FIX: Convert BGR to RGB (for model)
                mouth_crop_rgb = cv2.cvtColor(mouth_crop_resized, cv2.COLOR_BGR2RGB)
                
                # 4. FIX: NO manual normalization. Just convert type.
                # The model's Rescaling layer will handle normalization.
                mouth_crop_float = mouth_crop_rgb.astype(np.float32)
                
                # 5. Expand dimensions
                input_tensor = np.expand_dims(mouth_crop_float, axis=0)

                # --- 6. Make Prediction (FIX for stability) ---
                # Call model with training=False to ensure dropout/augmentation is off
                prediction = model(input_tensor, training=False)
                confidence = prediction.numpy()[0][0]

                # --- 7. Interpret Result ---
                if confidence > THRESHOLD:
                    prediction_text = f"YAWNING ({confidence*100:.0f}%)"
                    text_color = (0, 0, 255) # Red
                else:
                    prediction_text = f"Not Yawning ({(1-confidence)*100:.0f}%)"
                    text_color = (0, 255, 0) # Green

                cv2.rectangle(image, (x1, y1), (x2, y2), text_color, 2)
                break 

    # --- 8. Display the Result ---
    cv2.putText(image, prediction_text, (10, 30), FONT, 1, text_color, 2, cv2.LINE_AA)
    cv2.imshow('Yawn Detector Test', image)

    if cv2.waitKey(5) & 0xFF == 27: # Press 'ESC' to quit
        break

# --- 9. Cleanup ---
cap.release()
cv2.destroyAllWindows()