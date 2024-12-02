import cv2
import os
import numpy as np

# Initialize the LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Paths
dataset_path = "known_faces"
labels = []
faces = []

# Create a mapping for label to name
name_mapping = {}

label = 0  # Start from label 0

# Load the face cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iterate through each directory (person)
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    
    if os.path.isdir(person_folder):
        name_mapping[label] = person_name  # Map the label to the person's name

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            
            if image_name.endswith('.jpg'):
                img = cv2.imread(image_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

                # Detect faces in the image
                faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                for (x, y, w, h) in faces_detected:
                    # Crop the face from the image
                    face = gray[y:y+h, x:x+w]
                    
                    # Resize the face to a fixed size (optional, but recommended for consistency)
                    face = cv2.resize(face, (200, 200))
                    
                    faces.append(face)
                    labels.append(label)
                
        label += 1  # Increment label for the next person

print(f"Number of faces: {len(faces)}")
print(f"Number of labels: {len(labels)}")

# Train the LBPH recognizer
recognizer.train(faces, np.array(labels))

# Save the trained model
recognizer.save('trainer.yml')

# Save the name to label mapping
with open('name_mapping.txt', 'w') as f:
    for label, name in name_mapping.items():
        f.write(f'{label}: {name}\n')

print("Model trained and saved!")
