import pytest
import numpy as np
import cv2 as cv
import os


class TestTOpenCV(object):
   @pytest.mark.skip(reason="not working")
   def test_face_detection(self):
    print(os.getcwd())
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
    img = cv.imread('tests/resources/data/john_smith.jpeg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

   def test_face_detection2(self):
        filename = 'tests/resources/data/john_smith.jpeg'
        templatename = 'tests/resources/haarcascade_frontalface_default.xml'

        face_cascade = cv.CascadeClassifier(templatename)

        img = cv.imread(filename)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv.namedWindow('Face')
        cv.imshow('Face Detected!', img)
        cv.imwrite('tests/resources/john_smith_face_detected.jpg', img)
        # cv.waitKey(0)