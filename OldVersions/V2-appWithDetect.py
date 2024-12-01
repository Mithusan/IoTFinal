import logging
import time
import datetime
import io
import cv2
import socketserver
from http import server
from threading import Condition
from picamera2 import MappedArray, Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
import os
from ultralytics import YOLO
import numpy as np
from threading import Thread

colour = (0, 255, 0)
origin = (0, 30)
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1
thickness = 2
frameCount = 0

# Load YOLOv3-tiny model
model = YOLO("yolov3-tinyu.pt")  # Ensure the file is in the working directory

def detect_humans(frame):
    """Run YOLOv3-tiny inference on a frame and return detections."""
    # Convert the frame to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLO inference
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
    return detections

def human_detection_thread():
    """Thread to perform human detection and log results."""
    global frame
    while True:
        if frame is not None:
            detections = detect_humans(frame)
            for _, _, _, _, conf in detections:
                print(f"Human detected: {conf * 100:.2f}% confidence")

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
    detection_thread = Thread(target=human_detection_thread, daemon=True)
    detection_thread.start()

    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    server.serve_forever()

except Exception as e:
    print("Exception occurred ::: ", e)
    pass

finally:
    picam2.stop()
