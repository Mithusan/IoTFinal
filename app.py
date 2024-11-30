#!/usr/bin/python3

#afullagar@gmail.com blog.fullys.xyz

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

OUTPUTPATH = "/home/pi/html-php-scripts/mp4/"  # Unused now
CLIPLENGTH = 30

colour = (0, 255, 0)
origin = (0, 30)
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1
thickness = 2
frameCount = 0

# Removed frame count from timestamp
def apply_timestamp(request):
    timestamp = str(time.strftime("%Y-%m-%d %X"))
    with MappedArray(request, "main") as m:
        cv2.putText(m.array, timestamp, origin, font, scale, colour, thickness)

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

# Removed file saving and encoding to MP4 part
def h264_encode():
    while not encode_abort:
        now = datetime.datetime.now()
        try:
            # Create directory if it doesn't exist
            objectPath = OUTPUTPATH + now.strftime("%Y-%m-%d") + '/'
            if not os.path.exists(objectPath):
                os.makedirs(objectPath, mode=0o755, exist_ok=True)
        except Exception as e:
            print("Exception occurred creating the current date ::: ", e)
            pass

        picam2.pre_callback = apply_timestamp
        encoder2 = MJPEGEncoder()

        # No file saving portion here anymore

        time.sleep(CLIPLENGTH)

# Main streaming setup
try:
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (1980, 1020)}))
    output1 = StreamingOutput()
    encoder1 = MJPEGEncoder()
    picam2.start_recording(encoder1, FileOutput(output1))

    encode_abort = False
    encode_thread = Thread(target=h264_encode, daemon=False)
    encode_thread.start()

    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    server.serve_forever()

except Exception as e:
    print("Exception occurred ::: ", e)
    pass

finally:
    picam2.stop()
