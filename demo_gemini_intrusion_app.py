from flask import Flask, jsonify, Response, request, render_template_string
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import threading
import random
from datetime import datetime
import os
import time
import json
import geocoder
from werkzeug.utils import secure_filename
import requests
import base64
from flask_cors import CORS

app = Flask(__name__)
# CORS(app) 
CORS(app, resources={
    r"/start_detection": {"origins": "*"},
    r"/stop_detection": {"origins": "*"},
    r"/status": {"origins": "*"},
    r"/current_data": {"origins": "*"},
    r"/video_feed": {"origins": "*"},
    r"/toggle_mask_detection": {"origins": "*"},
    r"/update_roi": {"origins": "*"},
    r"/get_logs": {"origins": "*"},
    r"/get_image/*": {"origins": "*"}
})

# ThingsBoard Configuration
THINGSBOARD_URL = "http://43.205.64.196:8080/api/v1/hnFF4SXNSfk9S6UDkovi/telemetry"
CAMERA_FEED_URL = "http://192.168.1.122:8001/video_feed" 

# HTML template for testing interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Intrusion Detection Test Interface</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .panel { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
        .button { 
            background-color: #4CAF50; color: white; padding: 10px 15px; 
            border: none; border-radius: 4px; cursor: pointer; margin-right: 10px;
        }
        .button.stop { background-color: #f44336; }
        .button.safety { background-color: #2196F3; }
        .video-container { margin-top: 20px; }
        .logs { max-height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; }
        pre { background: #f4f4f4; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Intrusion Detection System Test</h1>
        
        <div class="panel">
            <h2>Detection Controls</h2>
            <button class="button" onclick="startDetection()">Start Detection</button>
            <button class="button stop" onclick="stopDetection()">Stop Detection</button>
            <button class="button safety" onclick="toggleSafetyDetection()">Toggle Safety Detection</button>
            <div id="status" style="margin-top: 10px;"></div>
        </div>
        
        <div class="panel">
            <h2>Live Feed</h2>
            <div class="video-container">
                <img id="videoFeed" src="{{ url_for('video_feed') }}" width="640" height="480">
            </div>
        </div>
        
        <div class="panel">
            <h2>Current Data</h2>
            <button class="button" onclick="getCurrentData()">Refresh Data</button>
            <pre id="currentData">Click "Refresh Data" to load</pre>
        </div>
        
        <div class="panel">
            <h2>Detection Logs</h2>
            <button class="button" onclick="getLogs()">Refresh Logs</button>
            <div class="logs" id="logs"></div>
        </div>
        
        <div class="panel">
            <h2>Test API Endpoints</h2>
            <p>You can also test these endpoints directly:</p>
            <ul>
                <li><a href="/start_detection" target="_blank">/start_detection</a></li>
                <li><a href="/stop_detection" target="_blank">/stop_detection</a></li>
                <li><a href="/toggle_safety_detection" target="_blank">/toggle_safety_detection</a></li>
                <li><a href="/current_data" target="_blank">/current_data</a></li>
                <li><a href="/get_logs" target="_blank">/get_logs</a></li>
                <li><a href="/status" target="_blank">/status</a></li>
            </ul>
        </div>
    </div>
    
    <script>
        function updateStatus(message) {
            document.getElementById('status').innerHTML = message;
        }
        
        function startDetection() {
            fetch('/start_detection')
                .then(response => response.json())
                .then(data => {
                    updateStatus(`Detection started: ${data.message}`);
                });
        }
        
        function stopDetection() {
            fetch('/stop_detection')
                .then(response => response.json())
                .then(data => {
                    updateStatus(`Detection stopped: ${data.message}`);
                });
        }
        
        function toggleSafetyDetection() {
            fetch('/toggle_safety_detection')
                .then(response => response.json())
                .then(data => {
                    updateStatus(`Safety detection: ${data.message}`);
                });
        }
        
        function getCurrentData() {
            fetch('/current_data')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('currentData').textContent = JSON.stringify(data, null, 2);
                });
        }
        
        function getLogs() {
            fetch('/get_logs')
                .then(response => response.json())
                .then(data => {
                    const logsDiv = document.getElementById('logs');
                    logsDiv.innerHTML = '';
                    data.forEach(log => {
                        const logEntry = document.createElement('div');
                        logEntry.style.borderBottom = '1px solid #eee';
                        logEntry.style.padding = '5px 0';
                        
                        const timestamp = document.createElement('strong');
                        timestamp.textContent = log.timestamp + ': ';
                        
                        const details = document.createElement('span');
                        details.textContent = `${log.class} (${log.confidence}) - ${log.restricted_area_violation} - ${log.person_count} people`;
                        
                        logEntry.appendChild(timestamp);
                        logEntry.appendChild(details);
                        logsDiv.appendChild(logEntry);
                    });
                });
        }
        
        // Refresh data every 5 seconds
        setInterval(getCurrentData, 5000);
    </script>
</body>
</html>
"""



class ObjectDetectorBackend:
    def __init__(self):
        self.model = None # Person detection model (YOLOv8n)
        self.mask_model = None # Safety/Mask detection model (safety.pt)
        self.cap = None
        self.class_colors = {
            'person': (0, 255, 0),       # Green for person
            'maskon': (0, 255, 0),        # Green for mask on
            'maskoff': (0, 0, 255)        # Red for mask off
            # Add other colors from safety.names if you plan to use them later
        }
        self.restricted_area_points = [
            (320, 0),    # Mid-top edge
            (640, 0),    # Top-right corner
            (640, 480),  # Bottom-right corner
            (320, 480)   # Mid-bottom edge
        ]
        self.csv_file = "data/detection_log.csv"
        self.object_entry_times = {} # Used for restricted area violation timing
        self.detection_running = False
        self.detection_thread = None
        self.image_save_dir = "captured_images"
        self.location = self.get_location()
        self.current_frame = None
        self.current_annotated_frame = None
        self.current_person_count = 0
        self.current_violations = [] # Restricted Area Violations
        self.current_mask_violations = [] # Mask Off Violations (list of dicts)
        self.frame_lock = threading.Lock()
        self.mask_detection_active = False
        self.last_status = None # For alert cooldown
        self.last_sent_time = 0
        self.cooldown = 7  # seconds
        self.person_classes = [] # Will be populated by YOLOv8n model
        self.mask_classes = [] # Will be populated from safety.names

        # --- Confidence Thresholds ---
        self.person_confidence_threshold = 0.5
        self.mask_confidence_threshold = 0.6 # Adjust as needed for mask detection

        # Create directories
        os.makedirs("data", exist_ok=True)
        os.makedirs(self.image_save_dir, exist_ok=True)

        # Initialize CSV
        if not os.path.exists(self.csv_file):
            pd.DataFrame(columns=[
                "timestamp", "class", "confidence", "location",
                "restricted_area_violation", "person_count",
                "image_path", "bounding_box", "violation_details",
                "mask_status", "image_base64" # Added image_base64
            ]).to_csv(self.csv_file, index=False)

    # --- Keep send_alert, get_location as they are ---
    def send_alert(self, intrusion_status, mask_violation_status):
        """Send alert to ThingsBoard with both intrusion and mask violation status"""
        img_base64 = ""
        with self.frame_lock:
            if self.current_annotated_frame is not None:
                _, buffer = cv2.imencode('.jpg', self.current_annotated_frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Count number of actual mask violations (people with mask off)
        num_mask_violations = len(self.current_mask_violations)

        data = {
            "url": { # Assuming this structure is expected by ThingsBoard
                "intrusion_detection": intrusion_status,
                "mask_violation": mask_violation_status, # True if any mask is off
                "url": CAMERA_FEED_URL,
                "person_count": self.current_person_count,
                "violations": len(self.current_violations), # Restricted area violations
                "mask_violation_count": num_mask_violations, # Count of people with mask off
                # "image": img_base64
            },
            "location": self.location,
            "mask_status": { # Additional mask details if needed
                "detection_active": self.mask_detection_active,
                "violations_list": self.current_mask_violations # Send details of each violation
            }
        }
        try:
            response = requests.post(THINGSBOARD_URL, json={"ts": int(time.time() * 1000), "values": data}, timeout=5)
            # Using print for now as URL is commented out
            # print(f"[!] Alert WOULD BE Sent (URL commented out): Intrusion={intrusion_status}, MaskViolation={mask_violation_status}, People={self.current_person_count}, MaskOffCount={num_mask_violations}")
            # print(f"    Data: {json.dumps(data)}") # Optionally print full data

            if response.status_code >= 200 and response.status_code < 300:
                print(f"[✓] Sent alert | Status Code: {response.status_code}")
            else:
                print(f"[x] Failed to send alert | Status Code: {response.status_code}, Response: {response.text}")

        except Exception as e:
            print(f"[x] Failed to send alert: {e}")

    def get_location(self):
        """Get approximate location using geocoder"""
        try:
            g = geocoder.ip('me')
            if g.ok:
                return {
                    "address": g.address, "lat": g.lat, "lng": g.lng,
                    "city": g.city, "country": g.country
                }
            else:
                print("Warning: Geocoder lookup failed.")
                return {"address": "Unknown", "lat": 0, "lng": 0, "city": "Unknown", "country": "Unknown"}
        except Exception as e:
            print(f"Error getting location: {e}")
            return {"address": "Unknown", "lat": 0, "lng": 0, "city": "Unknown", "country": "Unknown"}

    def load_models(self):
        """Load both person detection and mask detection models"""
        try:
            print("Loading YOLOv8n model for person detection...")
            self.model = YOLO("yolov8n.pt")
            self.person_classes = self.model.names # Get class names from the model
            print("YOLOv8n model loaded.")

            print("Loading safety.pt model for mask/safety detection...")
            self.mask_model = YOLO("safety.pt")
            print("safety.pt model loaded.")

            # Load mask class names from safety.names
            try:
                with open('safety.names', 'r') as f:
                    # Store as lowercase for consistent comparison
                    self.mask_classes = [line.strip().lower() for line in f.readlines()]
                print(f"Loaded mask classes: {self.mask_classes}")
                if 'maskon' not in self.mask_classes or 'maskoff' not in self.mask_classes:
                     print("Warning: 'maskon' or 'maskoff' not found in safety.names. Check the file content.")

            except FileNotFoundError:
                print("Error: safety.names not found. Mask detection will not work correctly.")
                self.mask_model = None # Disable mask model if names are missing
                return False

            print("All models loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    # --- Keep start_webcam, stop_webcam, draw_roi, is_near_restricted_area as they are ---
    def start_webcam(self, video_source=0):
        """Start webcam or video stream"""
        try:
            self.cap = cv2.VideoCapture(video_source)
            if not self.cap.isOpened():
                print("Error: Unable to access the video source.")
                return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print("Video source opened successfully.")
            return True
        except Exception as e:
            print(f"Error starting video source: {e}")
            return False

    def stop_webcam(self):
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
            self.cap = None
            print("Video source stopped.")

    def draw_roi(self, frame):
        """Draw restricted area polygon on frame"""
        if self.restricted_area_points:
            pts = np.array(self.restricted_area_points, np.int32).reshape((-1, 1, 2))
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 255)) # Red overlay
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame) # Apply transparency
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            # Find top-left point for text placement
            text_x = min(p[0] for p in self.restricted_area_points)
            text_y = min(p[1] for p in self.restricted_area_points)
            cv2.putText(frame, "RESTRICTED AREA", (text_x, text_y - 10 if text_y > 10 else text_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return frame

    def is_near_restricted_area(self, box):
        """Check if object's center is inside restricted area"""
        if self.restricted_area_points:
            pts = np.array(self.restricted_area_points, np.int32)
            x1, y1, x2, y2 = box
            object_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            # pointPolygonTest returns +1 (inside), 0 (on boundary), -1 (outside)
            return cv2.pointPolygonTest(pts, object_center, measureDist=False) >= 0
        return False # No restricted area defined

    # --- REMOVED detect_masks function ---

    def save_detection_data(self, frame, persons_data):
        """Save detection data for persons with violations and the annotated image"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        image_name = f"violation_{timestamp}.jpg"
        image_path = os.path.join(self.image_save_dir, image_name)

        # We already have the annotated frame ready in self.current_annotated_frame
        annotated_frame_to_save = None
        with self.frame_lock:
            if self.current_annotated_frame is not None:
                 annotated_frame_to_save = self.current_annotated_frame.copy()

        if annotated_frame_to_save is None:
             print("Warning: Cannot save image, annotated frame is None.")
             return None # Or maybe use the raw frame?

        # Save the annotated frame
        cv2.imwrite(image_path, annotated_frame_to_save)
        print(f"Saved violation image: {image_path}")

        # Convert image to base64 for CSV/JSON
        _, buffer = cv2.imencode('.jpg', annotated_frame_to_save)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Prepare data entries for CSV (one row per person *in this frame's snapshot*)
        log_entries = []
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        location_json = json.dumps(self.location)

        for person in persons_data:
            is_restricted = person['is_restricted']
            mask_status = person['mask_status'] # 'on', 'off', or 'unknown'
            # Log if there's *any* violation (restricted area OR mask off)
            # if is_restricted or mask_status == 'off': # Log only violations? Or all persons? Log all for now.
            entry = {
                "timestamp": current_time_str,
                "class": "person",
                "confidence": person['confidence'],
                "location": location_json,
                "restricted_area_violation": "Yes" if is_restricted else "No",
                "person_count": self.current_person_count, # Total count in frame
                "image_path": image_path, # Link all persons in frame to the same image
                "bounding_box": json.dumps(person['box']),
                "violation_details": json.dumps({"restricted": is_restricted, "mask_status": mask_status}), # More detailed
                "mask_status": mask_status,
                "image_base64": "..." # Avoid duplicating large base64 in CSV, maybe remove this column? Or store only for violations? Let's omit for CSV for size.
                # "image_base64": img_base64 # Uncomment if you really want base64 in CSV
            }
            log_entries.append(entry)

        # Append to CSV if there are entries
        if log_entries:
            df = pd.DataFrame(log_entries)
            df.to_csv(self.csv_file, mode='a', header=not os.path.exists(self.csv_file), index=False)
            print(f"Logged {len(log_entries)} person entries to CSV.")

        return image_path # Return path for potential use

    def _associate_masks_to_persons(self, person_boxes, mask_detections):
        """Associate mask detections ('maskon'/'maskoff') to person boxes."""
        person_mask_statuses = ["unknown"] * len(person_boxes) # Default status

        # Prioritize 'maskoff' detections
        for i, p_box in enumerate(person_boxes):
            px1, py1, px2, py2 = p_box
            person_center_x = (px1 + px2) // 2
            person_center_y = (py1 + py2) // 2

            found_mask_type = "unknown" # Track best mask found for this person

            for m_det in mask_detections:
                mx1, my1, mx2, my2 = m_det['box']
                mask_type = m_det['class_name'] # 'maskon' or 'maskoff'

                # Simple Check: Is mask center roughly within person box?
                # More robust: IoU or check if mask box is significantly contained within person box
                mask_center_x = (mx1 + mx2) // 2
                mask_center_y = (my1 + my2) // 2

                if px1 <= mask_center_x <= px2 and py1 <= mask_center_y <= py2:
                    if mask_type == 'maskoff':
                        found_mask_type = 'off'
                        break # Mask off overrides mask on for this person
                    elif mask_type == 'maskon':
                        found_mask_type = 'on'
                        # Don't break, keep checking for a 'maskoff'

            person_mask_statuses[i] = found_mask_type

        return person_mask_statuses

# <<< Keep the rest of the ObjectDetectorBackend class the same up to process_frame >>>

    def process_frame(self):
        """Process a single frame for person and mask detection"""
        if not self.cap or not self.model:
            # print("Process Frame: Cap or Model not ready.") # Less verbose
            return None, 0, []

        ret, frame = self.cap.read()
        if not ret:
            print("Error reading frame from video source.")
            time.sleep(1)
            # Basic reconnect attempt - consider more robust error handling
            self.stop_webcam()
            if not self.start_webcam():
                 print("Reconnection attempt failed.")
                 self.detection_running = False # Stop if reconnect fails
                 return None, 0, []
            return None, 0, [] # Return after starting webcam, process next frame

        # --- Stage 1: Person Detection ---
        person_results = self.model(frame, classes=[0], conf=self.person_confidence_threshold, verbose=False)
        persons_data = []
        person_boxes = []

        for result in person_results[0].boxes:
            conf = float(result.conf[0])
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            persons_data.append({
                "box": [x1, y1, x2, y2],
                "confidence": conf,
                "mask_status": "unknown",
                "is_restricted": False
            })
            person_boxes.append([x1, y1, x2, y2])

        # --- Stage 2: Mask Detection (Full Frame) ---
        mask_detections = [] # Store 'maskon'/'maskoff' detections FROM THIS FRAME
        if self.mask_detection_active and self.mask_model:
            target_class_indices = [i for i, name in enumerate(self.mask_classes) if name in ['maskon', 'maskoff']]
            if target_class_indices:
                mask_results = self.mask_model(frame, classes=target_class_indices, conf=self.mask_confidence_threshold, verbose=False)
                for result in mask_results[0].boxes:
                    class_id = int(result.cls)
                    # Basic check: ensure class_id is within bounds before accessing
                    if 0 <= class_id < len(self.mask_classes):
                        class_name = self.mask_classes[class_id] # 'maskon' or 'maskoff'
                        conf = float(result.conf[0])
                        x1, y1, x2, y2 = map(int, result.xyxy[0])
                        mask_detections.append({
                            "box": [x1, y1, x2, y2],
                            "class_name": class_name,
                            "confidence": conf
                        })
                    else:
                        print(f"Warning: Detected mask class_id {class_id} out of bounds for loaded classes.")

            # else: No warning needed if indices not found, handled in load_models

        # --- Stage 3: Associate Masks and Check Restricted Area ---
        associated_mask_statuses = ["unknown"] * len(persons_data)
        if self.mask_detection_active and mask_detections: # Use mask_detections from *this frame*
            associated_mask_statuses = self._associate_masks_to_persons(person_boxes, mask_detections)

        current_restricted_violations = []
        current_mask_off_violations = []

        # Need to track object entry times correctly even if no violation occurs yet
        # Use a copy of current times, update, then replace the main dict
        # (Or manage staleness - simpler for now is just checking restricted status)
        # active_object_ids = set() # Keep track of objects seen this frame

        for i, person in enumerate(persons_data):
            person['mask_status'] = associated_mask_statuses[i]
            person['is_restricted'] = self.is_near_restricted_area(person['box'])
            # person_id = tuple(person['box']) # Simple ID based on box
            # active_object_ids.add(person_id)

            if person['is_restricted']:
                violation_detail = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "box": person['box'], "confidence": person['confidence'], "type": "restricted_area"
                }
                current_restricted_violations.append(violation_detail)

                # Restricted area timing logic for saving
                person_id_for_timing = tuple(person['box'])
                entry_time = self.object_entry_times.get(person_id_for_timing, None)
                if entry_time is None:
                    self.object_entry_times[person_id_for_timing] = time.time() # Record first entry
                elif time.time() - entry_time > 2: # Persisted for > 2 sec
                    # Check if saving is needed (e.g., based on cooldown for this specific object?)
                    # For simplicity, let's trigger save but rely on overall alert cooldown
                    # self.save_detection_data(frame, persons_data) # Consider if needed here or just rely on alert logic
                    pass # Save logic moved outside loop or tied to alert cooldown

            # else: # Person NOT in restricted area
            #    person_id_for_timing = tuple(person['box'])
            #    if person_id_for_timing in self.object_entry_times:
            #        del self.object_entry_times[person_id_for_timing] # Remove if they leave the area

            if person['mask_status'] == 'off':
                 mask_violation_detail = {
                     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     "box": person['box'], "confidence": person['confidence'], "type": "mask_off"
                 }
                 current_mask_off_violations.append(mask_violation_detail)

        # Clean up old entries in object_entry_times (optional, prevents dict growing indefinitely if objects don't reappear)
        # current_time_for_cleanup = time.time()
        # self.object_entry_times = {pid: t for pid, t in self.object_entry_times.items() if pid in active_object_ids or current_time_for_cleanup - t < 60} # Keep active or recent


        # --- Stage 4: Update State and Annotate Frame ---
        annotated_frame = frame.copy()
        annotated_frame = self.draw_roi(annotated_frame)

        # 4a: Draw person boxes (with associated mask status and restricted marker)
        for person in persons_data:
            x1, y1, x2, y2 = person['box']
            conf = person['confidence']
            mask_status = person['mask_status']
            is_restricted = person['is_restricted']

            color = self.class_colors.get(f"mask{mask_status}", (0, 255, 0)) # Default green
            if mask_status == 'off': color = self.class_colors['maskoff']
            elif mask_status == 'on': color = self.class_colors['maskon']

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = f"Person {conf:.2f}"
            if self.mask_detection_active: label += f" [Mask: {mask_status}]"
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + 15
            cv2.putText(annotated_frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if is_restricted:
                violation_text_y = label_y - 20 if label_y - 20 > 10 else y2 + 15
                cv2.putText(annotated_frame, "RESTRICTED!", (x1, violation_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 4b: *** NEW: Draw RAW mask detection boxes IF mask detection is active ***
        if self.mask_detection_active and mask_detections:
            # print(f"Drawing {len(mask_detections)} raw mask boxes...") # Optional debug
            for m_det in mask_detections:
                mx1, my1, mx2, my2 = m_det['box']
                m_class = m_det['class_name']  # 'maskon' or 'maskoff'
                m_conf = m_det['confidence']
                # Use colors defined in class_colors, default if needed
                m_color = self.class_colors.get(m_class, (255, 255, 0)) # Default yellow

                # Draw the thinner rectangle for the mask itself
                cv2.rectangle(annotated_frame, (mx1, my1), (mx2, my2), m_color, 1) # Thickness 1

                # Draw label for the mask box (smaller font)
                m_label = f"{m_class} {m_conf:.2f}"
                mask_label_y = my1 - 5 if my1 - 5 > 5 else my1 + 10 # Slightly above box
                cv2.putText(annotated_frame, m_label, (mx1, mask_label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, m_color, 1) # Font size 0.4, thickness 1

        # 4c: Add overlay text (Counts, Location, Mask Status)
        person_count = len(persons_data)
        cv2.putText(annotated_frame, f"People: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Location: {self.location.get('city', 'Unknown')}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        overlay_status = "Mask Detection: ACTIVE" if self.mask_detection_active else "Mask Detection: INACTIVE"
        overlay_color = (0, 255, 255) if self.mask_detection_active else (128, 128, 128)
        cv2.putText(annotated_frame, overlay_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, overlay_color, 1)

        # Update shared state variables under lock
        with self.frame_lock:
            self.current_frame = frame
            self.current_annotated_frame = annotated_frame
            self.current_person_count = person_count
            self.current_violations = current_restricted_violations
            self.current_mask_violations = current_mask_off_violations

        # --- Stage 5: Send Alerts ---
        current_time = time.time()
        intrusion_detected = len(current_restricted_violations) > 0
        # Base mask violation on *associated* status on people for alerts, not just raw detections
        mask_violation_detected = len(current_mask_off_violations) > 0 # True if any person has mask off
        any_violation = intrusion_detected or mask_violation_detected
        current_overall_status = any_violation

        # Alerting Logic (same as before, based on overall status change or cooldown)
        if current_overall_status != self.last_status or \
           (current_overall_status and current_time - self.last_sent_time > self.cooldown):
            print(f"Alert Condition Met: Status Change ({self.last_status} -> {current_overall_status}) or Cooldown Passed ({current_time - self.last_sent_time:.1f}s > {self.cooldown}s). Sending alert.")
            self.send_alert(intrusion_detected, mask_violation_detected)
            # Trigger save operation when alert is sent
            if any_violation:
                 # Pass the latest persons_data containing all details for logging
                 self.save_detection_data(frame, persons_data)
            self.last_status = current_overall_status
            self.last_sent_time = current_time
        elif not current_overall_status and self.last_status is True:
            # Optionally send a 'clear' alert immediately when status goes from True to False
            print(f"Alert Condition Cleared: Status Change ({self.last_status} -> {current_overall_status}). Sending clear alert.")
            self.send_alert(False, False) # Send clear status
            self.last_status = False
            self.last_sent_time = current_time # Reset time for clear message

        return self.current_annotated_frame, self.current_person_count, self.current_violations


    # --- Keep run_detection, start_detection_process, stop_detection, toggle_mask_detection ---
    # Minor update to toggle_mask_detection message for clarity
    def run_detection(self):
        """Main detection loop"""
        if not self.load_models():
            self.detection_running = False # Ensure flag is false if models fail
            print("Failed to load models. Stopping detection.")
            return False
        if not self.start_webcam():
            self.detection_running = False # Ensure flag is false if webcam fails
            print("Failed to start webcam. Stopping detection.")
            return False

        # Already set self.detection_running = True in start_detection_process
        print("Detection loop started.")

        while self.detection_running:
            start_time = time.time()
            self.process_frame()
            end_time = time.time()
            # Optional: Print processing time
            # print(f"Frame processed in {end_time - start_time:.3f} seconds")

            # Control frame rate - sleep if processing is faster than desired FPS
            sleep_time = max(0, (1/30) - (end_time - start_time)) # Target ~30 FPS
            time.sleep(sleep_time)

            # Check flag again inside loop for responsiveness
            if not self.detection_running:
                 print("Detection flag turned off, exiting loop.")
                 break

        self.stop_webcam()
        print("Detection loop finished.")
        # Reset status for next run
        self.last_status = None
        self.last_sent_time = 0
        return True # Indicate clean stop

    def start_detection_process(self):
        """Start detection in a background thread"""
        if not self.detection_running:
            # Set flag immediately to prevent race condition
            self.detection_running = True
            print("Starting detection thread...")
            self.detection_thread = threading.Thread(target=self.run_detection, daemon=True)
            self.detection_thread.start()
            # Brief pause to allow thread to start and potentially fail early (e.g., model load)
            time.sleep(1)
            # Check if thread is alive and flag is still true (might have failed)
            if self.detection_thread.is_alive() and self.detection_running:
                return {"status": "success", "message": "Detection process starting"}
            else:
                 # It likely failed during initialization
                 self.detection_running = False # Reset flag
                 return {"status": "error", "message": "Detection process failed to start (check console logs)"}
        else:
            return {"status": "info", "message": "Detection already running"}

    def stop_detection(self):
        """Stop the detection process"""
        if self.detection_running:
            print("Stopping detection process...")
            self.detection_running = False # Signal the loop to stop
            if self.detection_thread and self.detection_thread.is_alive():
                 print("Waiting for detection thread to join...")
                 self.detection_thread.join(timeout=5) # Wait for thread to finish cleanly
                 if self.detection_thread.is_alive():
                     print("Warning: Detection thread did not stop gracefully.")
                 else:
                     print("Detection thread joined.")
            self.detection_thread = None
            # Ensure webcam is released if thread failed to do it
            if self.cap and self.cap.isOpened():
                 self.stop_webcam()
            return {"status": "success", "message": "Detection process stopped"}
        else:
            return {"status": "info", "message": "Detection not running"}

    def toggle_mask_detection(self):
        """Toggle mask detection on/off"""
        self.mask_detection_active = not self.mask_detection_active
        status = "ACTIVE" if self.mask_detection_active else "INACTIVE"
        print(f"Mask detection toggled to: {status}")
        return {"status": "success", "message": f"Mask detection set to {status}"}

    # --- Update get_current_detection_data ---
    def get_current_detection_data(self):
        """Get current detection data for API"""
        img_base64 = ""
        # Grab data under lock to ensure consistency
        with self.frame_lock:
            person_count = self.current_person_count
            restricted_violations = self.current_violations[:] # Make copies
            mask_violations = self.current_mask_violations[:] # Make copies
            mask_active = self.mask_detection_active
            annotated_frame = self.current_annotated_frame

            if annotated_frame is not None:
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "person_count": person_count,
            "restricted_area_violations": restricted_violations, # Renamed for clarity
            "mask_off_violations": mask_violations, # Renamed for clarity
            "location": self.location,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mask_detection_active": mask_active,
            "image_base64": img_base64 # Renamed for clarity
        }


# --- Flask Routes (Keep as they were, but ensure they call the correct methods) ---
# Initialize the detector outside the routes
detector_backend = ObjectDetectorBackend()

# app = Flask(__name__)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/')
def index():
    """Test interface for browser testing"""
    return render_template_string(HTML_TEMPLATE) # Ensure HTML uses updated data keys if needed

@app.route('/start_detection')
def start_detection_endpoint():
    result = detector_backend.start_detection_process()
    return jsonify(result)

@app.route('/stop_detection')
def stop_detection_endpoint():
    result = detector_backend.stop_detection()
    return jsonify(result)

@app.route('/status')
def get_status_endpoint():
    # Get status directly from the backend object
    return jsonify({
        "detection_running": detector_backend.detection_running,
        "location": detector_backend.location,
        "mask_detection_active": detector_backend.mask_detection_active
    })

@app.route('/current_data')
def get_current_data():
    data = detector_backend.get_current_detection_data()
    # Ensure keys match what the frontend/API consumer expects
    # Example: renaming keys if HTML template expects specific names
    # data['violations'] = data.pop('restricted_area_violations')
    # data['mask_violations'] = data.pop('mask_off_violations')
    # data['image'] = data.pop('image_base64')
    return jsonify(data)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        last_frame_time = 0
        min_interval = 1/30 # Max ~30 fps for the stream
        while True: # Loop controlled externally by client disconnecting
            if not detector_backend.detection_running:
                # Optionally, send a 'stopped' image or just break
                print("Video feed generator: Detection not running.")
                break # Stop generation if detection stops

            frame_to_send = None
            with detector_backend.frame_lock:
                if detector_backend.current_annotated_frame is not None:
                    frame_to_send = detector_backend.current_annotated_frame.copy()

            if frame_to_send is not None:
                current_time = time.time()
                if current_time - last_frame_time >= min_interval:
                    ret, buffer = cv2.imencode('.jpg', frame_to_send)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        last_frame_time = current_time
                    else:
                        print("Video feed generator: Error encoding frame.")
                else:
                    # Sleep briefly if trying to send faster than interval
                     time.sleep(min_interval / 5) # Sleep fraction of interval

            else:
                 # No frame available yet, wait briefly
                 # print("Video feed generator: No frame available, waiting...")
                 time.sleep(0.1)

        print("Video feed generator finished.")
    # response = Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    # response.headers.add('Access-Control-Allow-Origin', '*')
    # return response
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Renamed endpoint function to avoid conflict with backend method name
@app.route('/toggle_mask_detection')
def toggle_mask_detection_endpoint():
    result = detector_backend.toggle_mask_detection()
    return jsonify(result)


@app.route('/update_roi', methods=['POST'])
def update_roi():
    """Update restricted area points"""
    try:
        data = request.get_json()
        points = data.get('points')
        if points and isinstance(points, list) and len(points) >= 3:
            # Validate points are tuples/lists of 2 numbers
            valid_points = []
            for p in points:
                if isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(coord, (int, float)) for coord in p):
                     valid_points.append(tuple(map(int, p))) # Convert to int tuples
                else:
                     raise ValueError("Invalid point format")

            if len(valid_points) >= 3:
                 with detector_backend.frame_lock: # Access lock if modifying shared resource
                    detector_backend.restricted_area_points = valid_points
                 print(f"ROI updated to: {valid_points}")
                 return jsonify({"status": "success", "message": "ROI updated"})
            else:
                 return jsonify({"status": "error", "message": "Invalid points data - need at least 3 valid points"}), 400
        else:
             return jsonify({"status": "error", "message": "Invalid points data - must be a list of >= 3 points"}), 400
    except Exception as e:
        print(f"Error updating ROI: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/get_logs')
def get_logs():
    """Get detection logs"""
    try:
        if os.path.exists(detector_backend.csv_file):
            # Read last N lines for performance? Or provide pagination?
            # For now, read whole file. Be cautious with very large files.
            df = pd.read_csv(detector_backend.csv_file)
            # Convert NaN to None for JSON compatibility
            df = df.where(pd.notnull(df), None)
            return jsonify(df.to_dict(orient='records'))
        return jsonify([]) # Return empty list if file doesn't exist
    except Exception as e:
        print(f"Error getting logs: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/get_image/<filename>')
def get_image(filename):
    """Serve saved images"""
    try:
        safe_filename = secure_filename(filename)
        # Basic path traversal check (though secure_filename helps)
        if '..' in safe_filename or safe_filename.startswith('/'):
             return jsonify({"status": "error", "message": "Invalid filename"}), 400

        image_path = os.path.join(detector_backend.image_save_dir, safe_filename)
        if os.path.exists(image_path):
            # Use Flask's send_from_directory for better security and header handling
            # return send_from_directory(detector_backend.image_save_dir, safe_filename, mimetype='image/jpeg')
            # Manual reading if send_from_directory is not preferred:
             with open(image_path, 'rb') as f:
                 image_data = f.read()
             return Response(image_data, mimetype='image/jpeg')
        else:
             print(f"Image not found: {image_path}")
             return jsonify({"status": "error", "message": "Image not found"}), 404
    except Exception as e:
        print(f"Error getting image {filename}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# --- Main Execution ---
if __name__ == '__main__':
    # Ensure ThingsBoard URL is set correctly if uncommented in send_alert
    print("Starting Flask application...")
    # Set threaded=True for handling multiple requests (like video feed + API calls)
    # use_reloader=False is crucial when running background threads like the detection loop
    app.run(host='0.0.0.0', port=8001, debug=True, threaded=True, use_reloader=False)