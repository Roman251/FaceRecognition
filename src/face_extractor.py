import cv2

# xml file for detecting frontal-face
file = "../xml_files/haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(file)

def face_extractor(img):
    """
    Function detects faces and returns the cropped face
    If no face detected, it returns the input image
    """
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.05,
	                    minNeighbors=15, minSize=(30, 30),
	                    flags=cv2.CASCADE_SCALE_IMAGE)
    

    """
    scaleFactor  : How much the image size is reduced at each image scale.
                   (This means that this size of face is detected in the image if occuring. However, by rescaling the input image, you can resize a larger face towards a smaller one, making it detectable for the algorithm.)
    
    minNeighbors : This number determines the how much neighborhood is required to pass it as a face rectangle.
                   (Higher value results in less detections but with higher quality.)
    
    minSize      : A tuple of width and height (in pixels) indicating the window’s minimum size.
                   (Objects smaller than that are ignored)

    """
    # prints the coordinates of the bounding box enclosing the face
    # print(faces)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        # will create a bounding box in the original frame 
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

if __name__=='__main__':
    # access laptop video-camera
    # cv2.VideoCapture(1) for external video camera 
    video_capture = cv2.VideoCapture(0)

    while True:
        _, frame = video_capture.read()

        # _   : boolean variable that returns true if the frame is available
        # frame : image array
        
        face = face_extractor(frame)

        # display the frame with face detected
        cv2.imshow("Video", frame)

        # display the cropped_face
        # cv2.imshow("Video", face)
        
        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()