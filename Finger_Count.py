!pip install opencv-python
!pip install mediapipe

import cv2
import mediapipe as mp

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Finger landmark indices
fingerIndices = [8, 12, 16, 20]  # Tips of the index, middle, ring, and pinky fingers
thumbIndices = (4, 2)  # Thumb tip and thumb MCP

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from webcam")
        break

    # Convert the image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Access the detected hand landmarks
    multiLandMarks = results.multi_hand_landmarks

    if multiLandMarks:
        for handLms in multiLandMarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            h, w, _ = img.shape
            handPoints = [(int(lm.x * w), int(lm.y * h)) for lm in handLms.landmark]

            # Count fingers
            fingersUp = 0

            # Thumb
            if handPoints[thumbIndices[0]][0] > handPoints[thumbIndices[1]][0]:  # Comparing x-coordinates
                fingersUp += 1

            # Other fingers
            for i in fingerIndices:
                if handPoints[i][1] < handPoints[i - 2][1]:  # Comparing y-coordinates
                    fingersUp += 1

            # Display the count on the image
            cv2.putText(img, f'Fingers: {fingersUp}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the image
    cv2.imshow("Image", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
