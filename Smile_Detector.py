import cv2

#Face and Smile Classifiers.
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

#Grab a Webcam Feed.
webcam = cv2.VideoCapture(0)

while True:
    #Read current frame from Webcam.
    successful_frame_read, frame = webcam.read()

    #If there is an error, abort.
    if not successful_frame_read:
        break

    #Change to grayscale.
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces first.
    faces = face_detector.detectMultiScale(frame_grayscale)

    #Run smile detection with each of those faces.
    for (x, y, w, h) in faces:

        #Create a rectangle around the face.
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)

        #Make the face sub-image.
        the_face = frame[y:y+h, x:x+w]

        #Grayscale the face.
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        #Detect smiles in the face.
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)

        #Draw a rectangle around the smile.
        #for (x_, y_, w_, h_) in smile:
        #   cv2.rectangle(frame, (x_, y_), (x_+w_, y_+h_), (50, 50, 200), 4)

        #Label this face as smiling.
        if len(smiles) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))
    #Show the current frame.
    cv2.imshow('Smile Detector', frame)

    #Stop if Q key is pressed.
    key = cv2.waitKey(1)

#Clean Up.
webcam.release()
cv2.destroyAllWindows()
