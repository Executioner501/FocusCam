# FocusCam
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/Executioner501/FocusCam)

FocusCam is a comprehensive, real-time attention and distraction monitoring system built with Python. It leverages your webcam to analyze facial cues and environmental objects to determine if a user is focused, drowsy, or distracted. The system combines multiple AI models to provide a robust and nuanced assessment of user attention.

## Features

- **Hybrid Gaze Tracking**: Utilizes a powerful combination of MediaPipe's high-fidelity iris tracking for rule-based analysis and a custom-trained TensorFlow model for deep learning-based gaze direction classification (forward, up, down, left, right).
- **Yawn Detection**: Employs a custom CNN model, trained on the YawDD dataset, to specifically detect yawns as an indicator of drowsiness.
- **Phone Detection**: Integrates a YOLOv8 model to identify when a user is looking at a cell phone, a primary source of distraction.
- **Combined Attention Score**: Prioritizes and aggregates statuses from all detectors (phone > yawn > gaze) to provide a single, debounced, and easy-to-understand status: "PAYING ATTENTION" or "DISTRACTED".
- **Real-Time Visual Feedback**: Overlays status, diagnostic information, and targeting guides directly onto the webcam feed for immediate user feedback.

## How It Works

The core logic resides in `combined.py`, which establishes a multi-layered detection pipeline that runs in real-time on a webcam feed.

1.  **Frame Capture**: The system captures frames from the default webcam.
2.  **Face & Landmark Detection**: MediaPipe's Face Mesh is used to detect the face and its 478 landmarks, including precise iris, eye, and mouth coordinates.
3.  **Distraction Analysis (Prioritized)**:
    -   **Phone Detection**: The frame is first passed to a YOLOv8 model. If a "cell phone" is detected with sufficient confidence, the status is immediately set to "DISTRACTED: PHONE".
    -   **Yawn Detection**: If no phone is detected, the mouth region is cropped based on facial landmarks and fed into a custom-trained yawn detection model. If a yawn is detected, the status is set to "DISTRACTED: YAWNING".
    -   **Gaze Detection**: If neither a phone nor a yawn is detected, the system analyzes the user's gaze. It calculates the relative position of the irises within the eyes. This rule-based vector is combined with a prediction from a custom gaze classification model. If the final verdict is not "forward", the status is set to "DISTRACTED: LOOKING AWAY".
4.  **Status Display**: If none of the above distraction conditions are met, the status is "PAYING ATTENTION". The final status is debounced to prevent flickering and displayed clearly on the screen.

## Models

This repository includes several pre-trained models.

-   **Gaze Model (`gaze_model_1.keras`)**: A Keras CNN model trained on a custom dataset (`eye_dataset/`) of cropped eye images, created using the `gazefarm.py` script. The training process is documented in `GazeTester.ipynb`.
-   **Yawn Model (`yawn_model.h5`)**: A Keras CNN based on MobileNetV2, trained to classify mouth crops as "yawn" or "no_yawn". The data preprocessing and training pipeline, which uses the YawDD video dataset, is detailed in `NewYawn.ipynb`.
-   **Object Detection Model (`yolov8n.pt`)**: The standard YOLOv8-nano model from Ultralytics, used for its efficiency and accuracy in detecting common objects, including cell phones.

## Installation

To get started with FocusCam, follow these steps.

**Prerequisites:**
- Python 3.8+
- A connected webcam

**Setup:**

1.  Clone the repository to your local machine:
    ```bash
    git clone https://github.com/executioner501/FocusCam.git
    cd FocusCam
    ```

2.  Install the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the main application, execute the `combined.py` script.

```bash
python combined.py
```

A window will open showing your webcam feed with the attention status and diagnostic information overlaid. Press the `ESC` key to close the application.

### Individual Testers

You can test each component of the system individually:

-   **Gaze Detection**: `python newgaze.py`
-   **Yawn Detection**: `python yawn_tester.py`
-   **Phone Detection**: `python phone_tester.py`

### Data Collection

To collect your own eye images for retraining the gaze model, you can use the `gazefarm.py` script.

```bash
python gazefarm.py
```

Press `f`, `l`, `r`, `u`, `d` to label the direction you are looking (forward, left, right, up, down) and the script will automatically save cropped eye images to the `eye_dataset` directory.

### Contributors

  <a href="https://github.com/Executioner501/FocusCam/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=Executioner501/FocusCam" />
  </a>
