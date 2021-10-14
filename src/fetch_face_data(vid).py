# Fetch the data required for face recognition

import os
import cv2

import numpy as np
from face_extractor import face_extractor

# will become the target variable
name = input("Enter your name : ")

video_capture = cv2.VideoCapture(0)

file = "../xml_files/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(file)

frames  = []
outputs = []

error_count = 0

while True:

    _, frame = video_capture.read()

    cropped_face = face_extractor(frame)

    try:
        """
        resize funtion only works when images(faces) are detected
        throws an error when images are not detected
        """
        # resize image
        fix = cv2.resize(cropped_face, (100, 100))

        # convert to grayscale
        gray = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY)
    
    except Exception as e:
        error_count += 1
        print("Error", error_count)


    cv2.imshow("My Screen", frame)
    cv2.imshow("My Face",  gray)

    key = cv2.waitKey(1)

    # press q to exit
    if key == ord("q"):
        break

    # press c to fetch the data
    if key == ord("c"):
        frames.append(gray.flatten())
        outputs.append([name])

        print("data extraction completed")

data = np.hstack([np.array(outputs), np.array(frames)])

file_name = "../data/numpy_encodings.npy"

if os.path.exists(file_name):
    old  = np.load(file_name)
    data = np.vstack([old, data])

np.save(file_name, data)

video_capture.release()
cv2.destroyAllWindows()