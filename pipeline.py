import cv2
import mediapipe as mp
import math

# --- Setup ---

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV Webcam Capture
cap = cv2.VideoCapture(0) # 0 is the default webcam

# Define pixel size for cropping boxes (e.g., 50x50 pixels)
# You can tune these
MOUTH_BOX_SIZE = 50
EYE_BOX_SIZE = 30

# --- Main Loop ---

# Use the FaceDetection model
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # --- Image Processing ---
        
        # 1. Flip the image horizontally for a natural, mirror-like view
        image = cv2.flip(image, 1)

        # 2. Get image dimensions
        image_height, image_width, _ = image.shape

        # 3. Convert BGR image to RGB for MediaPipe
        #    MediaPipe models expect RGB images.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 4. Process the image and find faces
        results = face_detection.process(image_rgb)

        # --- Drawing and Cropping Logic ---

        if results.detections:
            for detection in results.detections:
                
                # A) Get Keypoints
                # MediaPipe returns normalized coordinates (0.0 to 1.0)
                # We need to convert them to pixel coordinates
                
                # MOUTH
                mouth_center_norm = mp_face_detection.get_key_point(
                    detection, mp_face_detection.FaceKeyPoint.MOUTH_CENTER)
                mouth_x = int(mouth_center_norm.x * image_width)
                mouth_y = int(mouth_center_norm.y * image_height)

                # LEFT EYE
                left_eye_norm = mp_face_detection.get_key_point(
                    detection, mp_face_detection.FaceKeyPoint.LEFT_EYE)
                left_eye_x = int(left_eye_norm.x * image_width)
                left_eye_y = int(left_eye_norm.y * image_height)

                # RIGHT EYE
                right_eye_norm = mp_face_detection.get_key_point(
                    detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)
                right_eye_x = int(right_eye_norm.x * image_width)
                right_eye_y = int(right_eye_norm.y * image_height)

                # B) Draw Rectangles (Visual Feedback)
                # We calculate the top-left (pt1) and bottom-right (pt2) corners
                
                # Mouth Box
                mouth_pt1 = (mouth_x - MOUTH_BOX_SIZE // 2, mouth_y - MOUTH_BOX_SIZE // 2)
                mouth_pt2 = (mouth_x + MOUTH_BOX_SIZE // 2, mouth_y + MOUTH_BOX_SIZE // 2)
                cv2.rectangle(image, mouth_pt1, mouth_pt2, (0, 255, 0), 2) # Green
                
                # Left Eye Box
                left_eye_pt1 = (left_eye_x - EYE_BOX_SIZE // 2, left_eye_y - EYE_BOX_SIZE // 2)
                left_eye_pt2 = (left_eye_x + EYE_BOX_SIZE // 2, left_eye_y + EYE_BOX_SIZE // 2)
                cv2.rectangle(image, left_eye_pt1, left_eye_pt2, (255, 0, 0), 2) # Blue

                # Right Eye Box
                right_eye_pt1 = (right_eye_x - EYE_BOX_SIZE // 2, right_eye_y - EYE_BOX_SIZE // 2)
                right_eye_pt2 = (right_eye_x + EYE_BOX_SIZE // 2, right_eye_y + EYE_BOX_SIZE // 2)
                cv2.rectangle(image, right_eye_pt1, right_eye_pt2, (255, 0, 0), 2) # Blue

                # --- THIS IS YOUR NEXT STEP ---
                # You can now crop these regions to feed into your models
                # Example:
                # crop_mouth = image[mouth_pt1[1]:mouth_pt2[1], mouth_pt1[0]:mouth_pt2[0]]
                # This 'crop_mouth' is what you would pass to your Yawn Detector model.
                # (You'd add error checking to make sure the box isn't outside the image)
                
        # --- Display the Result ---
        
        cv2.imshow('FocusCam Pipeline', image)

        # Exit loop by pressing 'ESC'
        if cv2.waitKey(5) & 0xFF == 27:
            break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()