import cv2 as cv
import numpy as np
import pygame.mixer as mixer  # Import the pygame.mixer module

# Load face cascade and eye cascade from haarcascades folder
face_cascade = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("haarcascades/haarcascade_eye.xml")

# Capture video from the webcam
video_capture = cv.VideoCapture(0)

# Initialize the mixer module
mixer.init()

# Load the audio file
audio_file = "audio/alert.wav"  # Adjust the path to your audio file
mixer.music.load(audio_file)

# Read all frames from the webcam
while True:
    ret, frame = video_capture.read()
    frame = cv.flip(frame, 1)  # Flip so that the video feed is not flipped, and appears mirror-like.
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Check if eyes are closed
    eyes_closed = False

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) == 0:
            eyes_closed = True

        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

    cv.imshow('Shailendra Singh', frame)

    # If eyes are closed, play the audio
    if eyes_closed:
        mixer.music.play()

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Finally, when video capture is over, release the video capture and destroyAllWindows
video_capture.release()
cv.destroyAllWindows()
