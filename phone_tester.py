import cv2
from ultralytics import YOLO
import math

# Load the pre-trained model
model = YOLO('yolov8n.pt')

# Get the class names the model knows
class_names = model.names

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1) # Flip for mirror view
    
    # Run YOLO prediction on the frame
    # This finds all objects in the frame
    results = model(frame, stream=True, verbose=False) # stream=True is more efficient

    # Loop through the detected objects
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get the class ID (e.g., 0 for 'person', 67 for 'cell phone')
            cls_id = int(box.cls[0])
            class_name = class_names[cls_id]
            
            # --- THIS IS THE KEY ---
            # We only care about the 'cell phone' class
            if class_name == 'cell phone':
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int
                
                # Get confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                
                # Draw the box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box
                cv2.putText(frame, f"Phone: {confidence}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Phone Detector Test", frame)

    if cv2.waitKey(5) & 0xFF == 27: # Press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()