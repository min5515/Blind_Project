import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
cascadepath = "haarcascade_frontalface_default.xml"
facecascade = cv2.CascadeClassifier(cascadepath) ;

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, im = cam.read()

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        Id, conf = recognizer.predict(gray[y:y + h, x:x + w])

        if conf < 70:
            if Id == 1:
                Id = "MINU MUKESH "
            elif Id == 2:
                Id = "Minu MUKESH"
            elif Id == 3:
                Id = "Anagha joseph"
            elif Id == 4:
                Id = "Sahela"
               
        else:
            Id = "MINU MUKESH "
        cv2.putText(im, str(Id), (x, y - 40), font, 1, (255, 255, 255), 3)

    im = cv2.resize(im, (600, 400))
    cv2.imshow('face', im)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()