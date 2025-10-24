import cv2
import os

# Input face ID from the user
face_id = input('Enter your ID: ')

# Initialize video capture from webcam
vid_cam = cv2.VideoCapture(0)

# Load the pre-trained Haar Cascade face detection model
face_detector = cv2.CascadeClassifier('Haarcascade_frontalface_default.xml')

# Initialize the count of face captures
count = 0

# Start video frame capture
while True:
    pwd = os.getcwd()
    
    # Capture video frame-by-frame
    _, image_frame = vid_cam.read()
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    # Loop over detected faces and process them
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Increment the face capture count
        count += 1
        
        # Save the captured face image in the dataset folder
        cv2.imwrite("dataset/img." + str(face_id) + '.' + str(count) + ".png", gray[y:y + h, x:x + w])
        
        # Display the frame with the detected face
        cv2.imshow('frame', image_frame)
    
    # Exit if 'q' is pressed or 50 images have been captured
    if cv2.waitKey(1002) & 0xFF == ord('q'):
        break
    elif count >= 100:
        print("Successfully captured 100 face images.")
        break

# Release the video capture object and close all OpenCV windows
vid_cam.release()
cv2.destroyAllWindows()