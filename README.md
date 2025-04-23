# Intrusion and Safety Mask Detection System

This project implements a real-time detection system using Python, Flask, OpenCV, and YOLOv8 models. It detects people, identifies intrusions into predefined restricted areas, detects the presence or absence of safety masks ('maskon'/'maskoff'), and provides a web interface for control and monitoring. Detected events can be logged, saved with annotated images, and sent as telemetry data to a ThingsBoard instance.

## Features

*   **Person Detection:** Detects people in the video feed using YOLOv8n.
*   **Restricted Area Monitoring:** Define polygonal restricted zones and detect when people enter them.
*   **Mask Detection:** Detects if people are wearing masks (`maskon`) or not (`maskoff`) using a custom-trained YOLO model (`safety.pt`).
*   **Web Interface:** Provides a Flask-based web UI for:
    *   Starting/Stopping the detection process.
    *   Viewing a live annotated video feed.
    *   Toggling mask detection on/off.
    *   Viewing current detection counts and violation status.
    *   Refreshing detection logs.
*   **API Endpoints:** Exposes RESTful APIs for controlling the system and retrieving data (detailed below).
*   **Data Logging:** Logs detection events (especially violations) to a CSV file (`data/detection_log.csv`).
*   **Image Saving:** Saves annotated image snapshots to `captured_images/` when violations occur.
*   **ThingsBoard Integration:** Sends detection status, counts, location, and image data as telemetry to a configured ThingsBoard instance for dashboards and alerting (configured in the backend code).


## Tech Stack

*   **Backend:** Python 3.x, Flask
*   **AI Models:** YOLOv8 (Ultralytics implementation)
    *   `yolov8n.pt` (for person detection)
    *   `safety.pt` (for mask/safety detection - requires `safety.names`)
*   **Core Libraries:**
    *   `opencv-python` (for video processing)
    *   `ultralytics` (for YOLO models)
    *   `numpy`
    *   `pandas` (for CSV logging)
    *   `requests` (for ThingsBoard integration)
    *   `geocoder` (for approximate location)
    *   `Werkzeug` (Flask dependency)
*   **Frontend:** HTML, CSS, JavaScript (via Flask template)

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/vaishnaveesithena/ai-intrusion-detection.git
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install flask opencv-python numpy pandas ultralytics geocoder requests Werkzeug
    ```
    *Note: `ultralytics` will install PyTorch and other necessary dependencies.*

4.  **Download/Place AI Models:**
    *   Ensure the following files are present in the project's root directory (or adjust paths in the script):
        *   `yolov8n.pt` (Pre-trained YOLOv8 Nano model)
        *   `safety.pt` (Your custom-trained safety/mask model)
        *   `safety.names` (Text file containing class names for `safety.pt`, one per line, e.g., `maskon`, `maskoff`)

5.  **Configure ThingsBoard (Optional):**
    *   If you intend to send data to ThingsBoard, open the main Python script (e.g., `app.py`).
    *   Locate the `send_alert` function.
    *   Find the commented-out `THINGSBOARD_URL` variable.
    *   Uncomment it and replace the placeholder URL with your actual ThingsBoard server address, port, and the **Device Access Token** for the target device.
        ```python
        # Example:
        THINGSBOARD_URL = "http://YOUR_THINGSBOARD_IP:8080/api/v1/YOUR_DEVICE_ACCESS_TOKEN/telemetry"
        ```

## Running the Application

1.  **Activate Virtual Environment (if used):**
    ```bash
    source venv/bin/activate
    ```

2.  **Run the Flask App:**
    ```bash
    python app.py # Or the name of your main Python script
    ```
    *   The application will start, typically listening on `http://0.0.0.0:8001`.

3.  **Access the Web Interface:**
    *   Open your web browser and navigate to `http://localhost:8001` or `http://<your-server-ip>:8001`.

4.  **Using the Interface:**
    *   Click "Start Detection" to begin processing.
    *   The "Live Feed" should show the annotated video.
    *   Use "Stop Detection" to halt the process.
    *   Use "Toggle Safety Detection" to enable/disable mask detection boxes and status.
    *   Use "Refresh Data" and "Refresh Logs" to view current status and historical events.

## API Endpoints

The backend exposes the following API endpoints:

| Method | Endpoint                    | Description                                                                 | Response Type |
| :----- | :-------------------------- | :-------------------------------------------------------------------------- | :------------ |
| `GET`  | `/`                         | Serves the HTML web interface.                                              | HTML          |
| `GET`  | `/start_detection`          | Starts the detection process.                                               | JSON          |
| `GET`  | `/stop_detection`           | Stops the detection process.                                                | JSON          |
| `GET`  | `/status`                   | Gets current status (running, location, mask detection active).             | JSON          |
| `GET`  | `/current_data`             | Gets latest counts, violations, location, and Base64 annotated image.       | JSON          |
| `GET`  | `/video_feed`               | Provides the MJPEG live annotated video stream.                             | MJPEG Stream  |
| `GET`  | `/toggle_mask_detection`    | Enables/Disables the mask detection feature.                                | JSON          |
| `POST` | `/update_roi`               | Updates the restricted area polygon points (requires JSON body).            | JSON          |
| `GET`  | `/get_logs`                 | Retrieves historical detection logs from the CSV file.                      | JSON Array    |
| `GET`  | `/get_image/<filename>`     | Retrieves a specific saved violation image file.                            | Image/JPEG    |

*(See the [API Documentation for Frontend](API_DOC.md) for detailed request/response formats)* --> [Optional: Create a separate file for the detailed API doc you generated earlier]

## Configuration

*   **Camera Source:** Modified in the `start_webcam` function (default is `0` for the default system webcam).
*   **Restricted Area:** Default points are set in `ObjectDetectorBackend.__init__`. Can be updated via the `/update_roi` API.
*   **Confidence Thresholds:** `person_confidence_threshold` and `mask_confidence_threshold` in `ObjectDetectorBackend.__init__` control detection sensitivity.
*   **Log/Image Paths:** `csv_file` and `image_save_dir` define storage locations.
*   **ThingsBoard Cooldown:** `cooldown` variable in `ObjectDetectorBackend.__init__` sets the minimum time between alerts sent to ThingsBoard.

## Directory Structure
├── app.py # Main Flask application script
├── requirements.txt # Python dependencies
├── safety.pt # YOLO model for mask/safety detection
├── safety.names # Class names for safety.pt
├── yolov8n.pt # YOLO model for person detection
├── data/ # Directory for logs
│ └── detection_log.csv # Log file (created automatically)
├── captured_images/ # Directory for saved violation images (created automatically)
├── templates/ # Optional: If HTML is in separate files
│ └── index.html
├── static/ # Optional: For CSS/JS files
│ └── style.css
├── venv/ # Virtual environment directory (if used)
└── README.md # This file


## Troubleshooting

*   **CORS Errors (ThingsBoard):** If accessing ThingsBoard data *from the frontend* fails, configure CORS settings in your `thingsboard.yml` file on the ThingsBoard server to allow your frontend's origin. See [ThingsBoard CORS Documentation](https://thingsboard.io/docs/user-guide/install/config/).
*   **Model Loading Errors:** Ensure model files (`.pt`, `.names`) are correctly placed and paths in the script are correct. Check if dependencies (PyTorch, ultralytics) installed correctly.
*   **Camera Access Error:** Verify the camera index in `start_webcam` is correct and the camera is not in use by another application. Check permissions.
*   **Slow Performance:** Detection speed depends on hardware (CPU/GPU) and model size. Consider using a smaller model or hardware acceleration (GPU) if needed. Ensure the `ultralytics` library installed the correct PyTorch version (CPU or GPU).
