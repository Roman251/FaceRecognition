import cv2
import numpy as np

from sklearn.svm import SVC

# load the data for training
data = np.load("../data/numpy_encodings.npy")

X = data[:, 1:].astype(int)
y = data[:, 0]

model = SVC()
model.fit(X, y)

video_capture = cv2.VideoCapture(0)

detector = cv2.CascadeClassifier("../xml_files/haarcascade_frontalface_default.xml")

while True:

    _, frame = video_capture.read()
    faces = detector.detectMultiScale(frame)

    for face in faces:
        x, y, w, h = face

        cropped_face = frame[y:y+h, x:x+w]

        fix = cv2.resize(cropped_face, (100, 100))
        gray = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY)

        output = model.predict([gray.flatten()])

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, str(output[0]), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)

    cv2.imshow('Video', frame)

    key  = cv2.waitKey(1)

    if key == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()