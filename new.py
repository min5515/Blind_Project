import os
import cv2
import numpy as np
from PIL import Image

# Create LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the Haar Cascade for face detection
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]  # Ignore hidden files
    faceSamples = []
    Ids = []

    for imagePath in imagePaths:
        try:
            print(f"Processing {imagePath}")
            
            # Convert image to grayscale using PIL
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            
            # Extract the ID from the filename
            Id = int(os.path.split(imagePath)[-1].split(".")[0])  # Assuming ID is before the file extension
            
            # Detect faces in the image
            faces = detector.detectMultiScale(imageNp)
            
            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y+h, x:x+w])
                Ids.append(Id)
        except Exception as e:
            print(f"Error processing {imagePath}: {e}")
    
    return faceSamples, Ids  # Corrected return statement

# Path to dataset
path = 'dataset'

# Get faces and IDs
faces, Ids = getImagesAndLabels(path)

# Train the recognizer
if len(faces) > 0 and len(Ids) > 0:
    recognizer.train(faces, np.array(Ids))
    print("Successfully Trained")

    # Save the trained model
    recognizer.write('trainer.yml')
else:
    print("No faces found or data is incomplete.")