import cv2
import pygame
import numpy as np
from gtts import gTTS
from twilio.rest import Client
import time

# Set up the Raspberry Pi and cameras

# Open both cameras
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

# Set the frames per second (FPS) for the video stream
fps = 30

# Set the dimensions of the captured frames
frame_width = 640
frame_height = 480

# Set the dimensions of the combined frame
combined_width = frame_width * 2
combined_height = frame_height

# Initialize the combined frame
combined = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

# Load the YOLOv4-tiny model
net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

# Initialize Pygame for playing the warning sound
pygame.init()

# Set Twilio API credentials
account_sid = 'your_account_sid'
auth_token = 'your_auth_token'
client = Client(account_sid, auth_token)

while True:
    # Capture a frame from each camera
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    # Check if the frames were successfully captured
    if not (ret0 and ret1):
        break

    # Resize the frames to the desired dimensions
    frame0 = cv2.resize(frame0, (frame_width, frame_height))
    frame1 = cv2.resize(frame1, (frame_width, frame_height))

    # Combine the frames side by side
    combined[:, :frame_width] = frame0
    combined[:, frame_width:] = frame1

    # Compute the disparity map using the stereoBM or stereoSGBM function from the cv2 library
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(frame0, frame1)

    # Detect obstacles using YOLOv4-tiny
    blob = cv2.dnn.blobFromImage(combined, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * combined_width)
                center_y = int(detection[1] * combined_height)
                w = int(detection[2] * combined_width)
                h = int(detection[3] * combined_height)
                x = center_x - w // 2
                y = center_y - h // 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Draw bounding boxes around detected objects
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = box
        cv2.rectangle(combined, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Check if there are any objects detected in the disparity map
    obstacle_detected = np.any(disparity > 25)

    # If an obstacle is detected, play a warning sound and send an SMS alert
    if obstacle_detected:
        # Load the warning sound and play it
        sound = pygame.mixer.Sound('warning_sound.wav')
        sound.play()

        # Convert the warning message to speech using Google Text-to-Speech (gTTS)
        message = "Warning! Obstacle detected in the road ahead!"
        speech = gTTS(text=message, lang='en', slow=False)

        # Save the speech as an MP3 file
        speech.save('warning_message.mp3')

        # Send an SMS alert using Twilio
        message = client.messages.create(
            body="Warning! Obstacle detected in the road ahead!",
            from_='your_twilio_phone_number',
            to='your_mobile_phone_number'
        )

        # Wait for 5 seconds before playing the next warning sound and sending the next SMS alert
        time.sleep(5)

    # Display the combined frame
    cv2.imshow('Combined Frame', combined)

    # Wait for a key press and check if the user pressed the 'q' key to quit the program
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
        
# Release the cameras and close all windows
cap0.release()
cap1.release()
cv2.destroyAllWindows()