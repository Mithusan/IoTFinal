import logging
import time
import datetime
import io
import cv2
import socketserver
from http import server
from threading import Condition, Thread
from picamera2 import MappedArray, Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
import os
from libcamera import controls
from ultralytics import YOLO

colour = (0, 255, 0)
origin = (0, 30)
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1
thickness = 2
frameCount = 0

# Load YOLO model (ensure you have the model file, e.g., yolov8n.pt)
model = YOLO("yolov8n.pt")  # Adjust path as needed

def detect_human(request):
    timestamp = str(time.strftime("%Y-%m-%d %X"))
    with MappedArray(request, "main") as m:
        frame = m.array
        # Run YOLO detection
        detection = model(frame)[0]
        
        nb_person = 0
        # Loop over the detections
        for box in detection.boxes:
            data = box.data.tolist()[0]
            accuracy = data[4]
            label = model.names.get(box.cls.item())
            
            # Filter out low-confidence detections
            if float(accuracy) < 0.7:
                continue
            else:
                if label == "person":
                    nb_person += 1  # Count the number of people detected
                
                # Draw bounding box and label on the frame
                xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), colour, 2)
                y = ymin - 15 if ymin - 15 > 15 else ymin + 15
                cv2.putText(frame, f"{label} {accuracy*100:.1f}%", (xmin, y), font, 0.5, colour, 2)

        # Add timestamp to the frame
        cv2.putText(frame, timestamp, origin, font, scale, colour, thickness)

PAGE = """\
<html>
<head>
<title>picamera2 MJPEG streaming demo</title>
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
                        
                    # Apply timestamp and human detection
                    detect_human(frame)
                    
                    # Send the processed frame for streaming
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

# Main streaming setup
try:
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (1280, 720)}))
    picam2.set_controls({"FrameRate": 60})
    
    output1 = StreamingOutput()
    encoder1 = MJPEGEncoder()
    picam2.start_recording(encoder1, FileOutput(output1))

    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    server.serve_forever()

except Exception as e:
    print("Exception occurred ::: ", e)
    pass

finally:
    picam2.stop()
