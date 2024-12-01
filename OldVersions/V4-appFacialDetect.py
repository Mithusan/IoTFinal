import logging
import datetime
import io
import cv2
import socketserver
from http import server
from threading import Condition, Thread, Event
from picamera2 import MappedArray, Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
import os
from ultralytics import YOLO
import RPi.GPIO as GPIO
import numpy as np

colour = (0, 255, 0)  # Green for bounding boxes
origin = (0, 30)
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1
thickness = 2
frameCount = 0
max_images = 2  # Maximum number of images to keep
motion_detected = False # Global flag to track motion detection status
saved_images = [] # List to track saved images
faces = []
labels = []

# Directories
output_directory = "captured_images"
known_faces_dir = "known_faces" 

# Ensure output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Models
model = YOLO("models/yolov3-tinyu.pt")  # Ensure the file is in the working directory
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

# Load the name to label mapping
name_mapping = {}
with open('name_mapping.txt', 'r') as f:
    for line in f:
        label, name = line.strip().split(': ')
        name_mapping[int(label)] = name

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(100, 100))
    
    name = "Unknown"
    confidence = 0
    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]  # Extract face region from the image
        label, confidence = recognizer.predict(face_region)  # Recognize the face

        name = name_mapping.get(label, "Unknown Person")

    return name, frame, confidence

def detect_humans(frame):
    # Detect faces first (to prioritize face detection)
    detected_person, face_detections, confidence = detect_faces(frame)
    
    # Run YOLO inference
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert the frame to RGB
    results = model.predict(img, conf=0.5)  # Adjust confidence threshold if needed

    # Extract detections for "person" class (Class ID: 0)
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:  # "person" class
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                conf = box.conf[0]  # Confidence score
                detections.append((x1, y1, x2, y2, conf))

                # Draw the bounding box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)
                label = f"Person: {conf * 100:.2f}%"
                cv2.putText(frame, label, (x1, y1 - 10), font, scale, colour, thickness)
    return detections, frame, detected_person, confidence

def detection_thread():
    global frame

    while True:
        if frame is not None:
            # Detect both humans (YOLO) and faces (OpenCV)
            detections, detected_frame, detected_person, confidence = detect_humans(frame)

            if detected_person:
                print(f"Face detected: {detected_person}, {confidence}% confidence")


            for _, _, _, _, conf in detections:
                print(f"Human detected: {conf * 100:.2f}% confidence")

            # Save the frame with detections if a human is detected
            if detections:
                img = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(output_directory, f"detected_{timestamp}.jpg")
                cv2.imwrite(filename, img)
                saved_images.append(filename)
                print(f"Saved frame as {filename}")

                # If the number of saved images exceeds the limit, delete the oldest
                if len(saved_images) > max_images:
                    oldest_image = saved_images.pop(0)
                    os.remove(oldest_image)
                    print(f"Deleted old image: {oldest_image}")

PAGE = """\
<html>
<head>
<title>TrueDetect</title>
</head>
<body>
<img src="stream.mjpg"/>
</body>
</html>
"""

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()

            try:
                while True:
                    with output1.condition:
                        output1.condition.wait()
                        frame = output1.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')

            except Exception as e:
                logging.warning('Removed streaming client %s: %s', self.client_address, str(e))

        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

# Initialize a global variable for the detection thread
frame = None

try:
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (1280, 720)}))
    picam2.set_controls({"FrameRate": 60})
    
    # Update the frame variable for detection
    def update_frame(request):
        global frame
        with MappedArray(request, "main") as m:
            frame = m.array.copy()

    picam2.pre_callback = update_frame  # Set callback to capture frames

    output1 = StreamingOutput()
    encoder1 = MJPEGEncoder()
    picam2.start_recording(encoder1, FileOutput(output1))

    # Start the human detection thread
    detection_thread = Thread(target=detection_thread, daemon=True)
    detection_thread.start()

    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    server.serve_forever()

except Exception as e:
    print("Exception occurred ::: ", e)
    pass

finally:
    picam2.stop()
    GPIO.cleanup()  # Clean up GPIO when done
