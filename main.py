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
import time
import BlynkLib
import socket
from flask import Flask, send_from_directory

# Blynk Authentication
BLYNK_AUTH = '1Cgsl73SNoP2p0cisduA4kv0GAjqYW4K'
BLYNK_TEMPLATE_ID = "TMPL24s7sajmq"
BLYNK_TEMPLATE_NAME = "TrueDetect"

blynk = BlynkLib.Blynk(BLYNK_AUTH, server='blynk.cloud', port=80)

colour = (0, 255, 0)  # Green for bounding boxes
origin = (0, 30)
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1
thickness = 2
frameCount = 0


max_images = 6  # Maximum number of images to keep
recent_images = []
saved_images = [] # List to track saved images
faces = []
notification_log=[]
ip_address = os.popen('hostname -I').read().strip()
app = Flask(__name__) # Flask Server Setup for Image Hosting
suppress_face_notifications = False
detection_measure = 0 # 0 for noone, 1 for person, 2 for face
motion_detected_event = Event()

# Setup GPIO for motion sensor
GPIO.setmode(GPIO.BCM)
MOTION_PIN = 17  # Pin connected to the HC-SR501 sensor
GPIO.setup(MOTION_PIN, GPIO.IN)

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

# Blynk Pins
V_PIN_STREAM = 1
V_PIN_LOG = 2
V_PIN_FACENOTI = 3

# Blynk Integration
def run_blynk():
    while True:
        blynk.run()

# Create a new thread for Blynk
blynk_thread = Thread(target=run_blynk, daemon=True)
blynk_thread.start()

def send_notification(message):    
    blynk.log_event("detected",message)  # Send a push notification

@blynk.VIRTUAL_WRITE(V_PIN_FACENOTI)
def toggle_face_notifications(value):
    global suppress_face_notifications
    print(int(value[0]))
    if int(value[0]) == 1:  # Switch is turned on
        suppress_face_notifications = True
        print("Face notifications suppressed.")
    else:  # Switch is turned off
        suppress_face_notifications = False
        print("Face notifications enabled.")

# Detection
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(100, 100))
    
    detected_name = None
    confidence = 0
    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]  # Extract face region from the image
        label, confidence = recognizer.predict(face_region)  # Recognize the face

        if (confidence) < 55:  # Convert LBPH's confidence score to match the threshold logic
            detected_name = name_mapping.get(label, "Unknown Person")
            print(f"Confidence {confidence} - Good")
        else:
            print(f"Low Confidence {confidence} - Skipped")
            detected_name = "Unknown"

    return detected_name, frame, confidence

def detect_humans(frame):
    # Detect faces first (to prioritize face detection)
    detected_person, face_detections, confidence = detect_faces(frame)
    conf=0
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
    return detections, frame, detected_person, conf

def detection_thread():
    global frame
    global detection_measure

    while True:
        motion_detected_event.wait()

        if frame is not None:
            # Detect both humans (YOLO) and faces (OpenCV)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            detections, detected_frame, detected_person, confidence = detect_humans(frame)
            for _, _, _, _, conf in detections:
                detection_measure=1
                detection_message = f"Human detected - {conf * 100:.2f}% confidence - Time:{current_time} !"


            if detected_person:
                if detected_person == "Unknown":
                    detection_measure=1
                    detection_message = f"Unknown Person - Time:{current_time} !"

                else:
                    detection_measure=2
                    detection_message = f"Face detected-{detected_person} - Time:{current_time} !"
            
            if suppress_face_notifications :
                print("Notification for face skipped due to switch state.")
                
            if detection_measure==2 and not suppress_face_notifications:
                print(detection_message)
                send_notification(f"{detected_person} is at the Front Door at Time:{current_time}")  # Send a push notification
            if detection_measure==1:
                print(detection_message)
                send_notification(f"Someone is at the Front Door at Time:{current_time}")  # Send a push notification

# Function to handle motion detection using the HC-SR501 sensor
def motion_detection_thread():
    global motion_detected_event
    while True:
        if GPIO.input(MOTION_PIN) == GPIO.HIGH:
            print("Motion detected!")
            motion_detected_event.set()  # Signal the detection thread to start
            time.sleep(5)  # Delay to avoid re-triggering too quickly
        else:
            motion_detected_event.clear()  # Stop detection when no motion
        time.sleep(0.1)  # Check for motion at intervals

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

    stream_url = f"http://{ip_address}:8000/stream.mjpg"
    blynk.set_property(V_PIN_STREAM, "urls", stream_url)

    #Start motion detection thread
    motion_thread = Thread(target=motion_detection_thread, daemon=True)
    motion_thread.start()

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