"""
Authors: Tomasz Rybczyński, Filip Marcoń

In order to be able to run script with this game you will need:
Python at least 3.8
opencv-python, pygame, dlib, numpy, scipy

To run script you need to run command "python3 ads.py"

This app tracks your eyes and plays a siren sound when you close your eyes for too long. Forces user to watch the ad.
"""

import cv2
import pygame
import dlib
import numpy as np
from scipy.spatial import distance as dist

# Load the Haar cascade xml files for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the dlib facial landmarks predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Threshold for EAR below which we will play the sound
EAR_THRESHOLD = 0.2


# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize pygame for playing sound
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=4096)
alert_sound = pygame.mixer.Sound('alert.mp3')

# Variable to keep track of whether the sound is currently playing
sound_playing = False

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        rect = dlib.rectangle(x, y, x+w, y+h)
        shape = predictor(gray, rect)

        shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)])

        # Get the coordinates of the left and right eyes
        leftEye = shape[36:42]
        rightEye = shape[42:48]

        # Calculate the EAR for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Average the EAR scores for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # If the eye is closed and the sound is not already playing, play the alert sound
        if ear < EAR_THRESHOLD and not sound_playing:
            alert_sound.play()
            sound_playing = True
            
        # If the eye is open and the sound is playing, stop the sound
        elif ear >= EAR_THRESHOLD and sound_playing:
            alert_sound.stop()
            sound_playing = False

        # If the sound is playing, draw the text on the frame
        if sound_playing:
            cv2.putText(frame, 'WATCH THE AD', (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()