import os
import cv2
import numpy as np
from PIL import Image

# Ensure you have installed opencv-contrib-python
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Path to the Haar Cascade XML file
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    # Check if the dataset path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} does not exist. Please check your dataset directory.")

    # Get all image paths in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    faceSamples = []
    Ids = []

    # Loop through all image paths and extract face data
    for imagePath in imagePaths:
        try:
            print(f"Processing {imagePath}")
            # Load the image and convert it to grayscale
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')

            # Extract the ID from the filename
            filename = os.path.split(imagePath)[-1]
            Id = int(filename.split(".")[1])

            # Detect faces in the image
            faces = detector.detectMultiScale(imageNp)

            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y + h, x:x + w])
                Ids.append(Id)

        except Exception as e:
            print(f"Error processing {imagePath}: {e}")

    return faceSamples, Ids

# Directory containing the dataset
dataset_path = 'dataset'

# Retrieve face samples and their corresponding IDs
faces, Ids = getImagesAndLabels(dataset_path)

# Train the recognizer
recognizer.train(faces, np.array(Ids))
print("Successfully trained the recognizer.")

# Save the trained model
recognizer.save('trainer.yml')
print("Model saved as 'trainer.yml'.")