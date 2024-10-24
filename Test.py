import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

# Initialize MediaPipe Hand class
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Setup the hands module
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Pycaw setup for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get the volume range (min: -65, max: 0)
min_vol = volume.GetVolumeRange()[0]
max_vol = volume.GetVolumeRange()[1]

# Variable to track whether volume is locked
volume_locked = False

# Function to check if the hand is open or closed
def is_hand_closed(hand_landmarks, image_shape):
    """
    Returns True if the hand appears closed (fist), and False if the hand is open.
    """
    # Calculate the distance between the tip of the pinky and the base of the hand (wrist)
    h, w, _ = image_shape
    wrist = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * w,
                      hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * h])
    pinky_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * w,
                          hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * h])

    # If the distance is short, the hand is likely closed (fist)
    distance = np.linalg.norm(pinky_tip - wrist)
    return distance < 100  # Threshold can be adjusted based on your testing

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Flip the image horizontally for a mirror effect
    image = cv2.flip(image, 1)

    # Process the image and detect hands
    results = hands.process(image)

    # Convert the image back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        # Track if left and right hands are detected
        left_hand = None
        right_hand = None

        # Classify the hands and detect landmarks
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Hand label (Left or Right)
            label = handedness.classification[0].label

            if label == 'Left':
                left_hand = hand_landmarks
            elif label == 'Right':
                right_hand = hand_landmarks

            # Draw hand landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Check if left hand is detected and whether it is open or closed
        if left_hand:
            if is_hand_closed(left_hand, image.shape):
                volume_locked = True
                cv2.putText(image, "Volume Locked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                volume_locked = False
                cv2.putText(image, "Volume Unlocked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Only adjust volume if the right hand is detected and volume is not locked
        if right_hand and not volume_locked:
            # Get the coordinates of the index finger tip and thumb tip
            index_finger_tip = right_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = right_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Get the height and width of the image
            h, w, c = image.shape

            # Convert the normalized landmarks to pixel coordinates
            index_finger_coords = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
            thumb_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))

            # Draw circles at the index finger tip and thumb tip
            cv2.circle(image, index_finger_coords, 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(image, thumb_coords, 10, (0, 255, 0), cv2.FILLED)

            # Calculate the Euclidean distance between the index finger tip and thumb tip
            distance = np.linalg.norm(np.array(index_finger_coords) - np.array(thumb_coords))

            # Normalize the distance to a range for controlling the volume
            volume_range = np.interp(distance, [30, 300], [min_vol, max_vol])

            # Set the system volume based on the distance
            volume.SetMasterVolumeLevel(volume_range, None)

            # Display the volume percentage on the screen
            cv2.putText(image, f'Volume: {int(np.interp(volume_range, [min_vol, max_vol], [0, 100]))}%', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the image with the landmarks
    cv2.imshow('Hand Detection and Volume Control', image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
