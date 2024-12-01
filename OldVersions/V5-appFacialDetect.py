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
import openface 
from openface import AlignDlib
from openface import TorchNeuralNet
import glob
from scipy.spatial.distance import cosine

colour = (0, 255, 0)  # Green for bounding boxes
origin = (0, 30)
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1
thickness = 2
frameCount = 0
max_images = 2  # Maximum number of images to keep
motion_detected = False # Global flag to track motion detection status
saved_images = [] # List to track saved images
detection_timeout_event = Event() # Event for controlling detection thread timeout
known_embeddings = {}

# Directories
output_directory = "captured_images"  # Directory to save captured frames with detections
known_faces_folder = "known_faces"

if os.path.exists(output_directory):
    # Remove all files in the directory
    for filename in os.listdir(output_directory):
        file_path = os.path.join(output_directory, filename)
        try:
            os.remove(file_path)  # Remove file
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")
else:
    os.makedirs(output_directory)

# Models
model = YOLO("yolov3-tiny.pt")
align = AlignDlib('shape_predictor_68_face_landmarks.dat')
net = TorchNeuralNet('nn4.small2.v1.t7')

def load_known_faces():
    global known_embeddings
    for image_path in glob.glob(os.path.join(known_faces_folder, "*.jpg")):
        base_name = os.path.basename(image_path)
        name = base_name.split('_')[0]

        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face and compute embedding
        det = align.getLargestFaceBoundingBox(rgb_image)
        if det is not None:
            landmarks = align.findLandmarks(rgb_image, det)
            embedding = net.forward(rgb_image, landmarks)
            
            # If this person already exists in the dictionary, append the new embedding
            if name in known_embeddings:
                known_embeddings[name].append(embedding)
            else:
                known_embeddings[name] = [embedding]  # Initialize with a list
        else:
            print(f"No face detected in {image_path}")

def compute_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

def detect_faces(frame):
    global known_embeddings
    
    # Convert the frame to RGB (OpenFace works in RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces using dlib's aligner
    dets = align.getAllFaceBoundingBoxes(rgb_frame)
    
    matches = []
    for det in dets:
        # Get the facial landmarks for the detected face
        landmarks = align.findLandmarks(rgb_frame, det)
        
        # Perform face recognition
        face_embedding = net.forward(rgb_frame, landmarks)
        
        # Compare with known embeddings
        best_match = None
        best_similarity = 0.0
        for name, embeddings in known_embeddings.items():  # Iterate over each person's embeddings
            for known_embedding in embeddings:
                similarity = compute_similarity(face_embedding, known_embedding)
                if similarity > best_similarity and similarity > 0.6:  # Similarity threshold
                    best_match = name
                    best_similarity = similarity
        
        if best_match:
            matches.append((best_match, best_similarity))
            # Draw bounding box and label
            x1, y1, x2, y2 = det.left(), det.top(), det.right(), det.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)
            label = f"{best_match}: {best_similarity * 100:.2f}%"
            cv2.putText(frame, label, (x1, y1 - 10), font, scale, colour, thickness)
    return matches

def detect_humans(frame):
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
    return detections, frame

def detection_thread():
    global frame

    while True:
        if frame is not None:
            # Detect both humans (YOLO) and faces (OpenCV)
            detections, detected_frame = detect_humans(frame)

            for _, _, _, _, conf in detections:
                print(f"Human detected: {conf * 100:.2f}% confidence")
                recognized_faces = detect_faces(detected_frame)

                if recognized_faces:
                    print(f"Face detected")

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
    load_known_faces()

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
