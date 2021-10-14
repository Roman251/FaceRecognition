import os
import cv2
import numpy as np

from face_extractor import face_extractor

file_name = "../data/numpy_encodings.npy"
name = input("Enter the name : ")

# load the image
path = '../input.jpg'
image = cv2.imread(path)

# SELECT SPECIFIC FACE FROM MULTIPLE DETECTED FACES 
""" 
file = "../xml_files/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(file)

faces = face_cascade.detectMultiScale(image, scaleFactor=1.05,
	                    minNeighbors=15, minSize=(30, 30),
	                    flags=cv2.CASCADE_SCALE_IMAGE)
        
# Crop all faces found

print(len(faces))

# create bounding box around second image 
for (x,y,w,h) in [faces[1]]:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)

cv2.imshow('Image', image)
cv2.waitKey(0)        

"""

# will return the left most face
cropped_face = face_extractor(image)
cv2.imshow('Image', cropped_face)
cv2.waitKey(0)

# image processing
fix  = cv2.resize(cropped_face, (100, 100))
gray = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY)

frames  = []
outputs = []

frames.append(gray.flatten())
outputs.append([name])

data = np.hstack([np.array(outputs), np.array(frames)])

if os.path.exists(file_name):
    old  = np.load(file_name)
    data = np.vstack([old, data])

np.save(file_name, data)

