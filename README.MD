# TrueDetect - IoT Enabled Smart Doorbell
The TrueDetect Smart Doorbell is designed to improve home security by combining motion detection with advanced machine learning techniques for more accurate identification of people and objects. Unlike traditional motion sensors that simply detect any movement, TrueDetect uses real-time object detection to distinguish between humans and other types of movement, significantly reducing the number of false alarms. By incorporating the YOLO3 (You Only Look Once) model for detecting objects and OpenCV's Haar Cascade for facial recognition, the system can tell the difference between familiar faces and strangers. This allows it to send precise notifications to users, keeping them informed only when necessary. The doorbell also streams live video to a mobile app via the Blynk platform, so users can see what's happening in real time, wherever they are. The main goal of this project is to create a smart doorbell camera that not only detects human presence but also enhances security by providing accurate, timely notifications and live video feeds directly to a user’s phone. With this system, users can have greater peace of mind, knowing they are always connected to what's happening at their front door.

## How to Run
1. **Upload Images to "known_faces" to Train Model for Facial Recognition**

2. **Run the Model Training Program**
   ```bash
   python3 LBPHTrain.py
   ```
3. **In main.py, replace "BLYNK_AUTH" with your Blynk Authentication Code**

3. **Run the Main Program**
   ```bash
   python3 main.py
   ```
