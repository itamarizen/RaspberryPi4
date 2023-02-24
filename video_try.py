import cv2
import pygame
import numpy as np
from gtts import gTTS
from twilio.rest import Client
import time

# Set up the demo video
cap = cv2.VideoCapture('demo.mp4')

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
    # Capture a frame from the demo video
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        break

    # Resize the frame to the desired dimensions
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Combine the frame with a blank space on the right
    combined[:, :frame_width] = frame
    combined[:, frame_width:] = 0

    # Compute the disparity map using the stereoBM or stereoSGBM function from the cv2 library
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(frame, frame)

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
        cv2.rectangle(combined, (x, y), (x+w, y+h), (0,0,255), 2)

    # Display the combined frame with bounding boxes
    cv2.imshow('Combined', combined)
    cv2.waitKey(1)

    # Play a warning sound and send text message if obstacles are detected
    if len(indices) > 0:
    # Play warning sound
        pygame.mixer.music.load('warning.wav')
        pygame.mixer.music.play()

    # Send text message
    message = client.messages.create(
        to='+1234567890',  # replace with your phone number
        from_='+0987654321',  # replace with your Twilio phone number
        body='Obstacle detected! Please be careful.')
    print(f'Text message sent to {message.to}')

# Display the combined frame with bounding boxes and disparity map
cv2.imshow('Combined', combined)
cv2.imshow('Disparity', disparity)

# Wait for the specified amount of time before capturing the next frame
key = cv2.waitKey(int(1000/fps))

# Press 'q' to quit
if key == ord('q'):
    break

# Release the resources
cap.release()
cv2.destroyAllWindows()

