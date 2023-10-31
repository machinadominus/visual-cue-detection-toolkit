import cv2
import dlib
import numpy as np
import mediapipe as mp
from fer import FER

# Initialize the FER model for emotion detection
fer_detector = FER()

# Load Haar Cascade classifiers for face and eye detection
# Load Haar Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier('C:/Users/sohail/Downloads/srp-main/srp-main/pretrained/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/sohail/Downloads/srp-main/srp-main/pretrained/haarcascade_eye.xml')

# Initialize dlib's face detector and facial landmark predictor for blink and gaze detection
detector_blink = dlib.get_frontal_face_detector()
predictor_blink = dlib.shape_predictor('C:/Users/sohail/Downloads/srp-main/srp-main/pretrained/shape_predictor_68_face_landmarks.dat')
detector_gaze = dlib.get_frontal_face_detector()
predictor_gaze = dlib.shape_predictor('C:/Users/sohail/Downloads/srp-main/srp-main/pretrained/shape_predictor_68_face_landmarks.dat')

# Rest of your code...

# Initialize some variables for blink detection
blink_counter = 0
frame_counter = 0
blink_rate = 0.0
blink_rate_display = "Blink Rate: {:.2f}".format(blink_rate)

# Function to calculate the eye aspect ratio (EAR) for blink detection
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm (eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to determine gaze direction
def determine_gaze_direction(left_eye_center, right_eye_center):
    eye_angle = np.arctan2(
        left_eye_center[1] - right_eye_center[1],
        left_eye_center[0] - right_eye_center[0]
    )
    eye_angle_deg = np.degrees(eye_angle)
    
    if abs(eye_angle_deg) < 10:
        gaze_direction = "Center"
    elif eye_angle_deg < -10:
        gaze_direction = "Left"
    else:
        gaze_direction = "Right"
    
    return gaze_direction

# Initialize video capture (you can change the parameter to your camera index)
cap = cv2.VideoCapture(0)

# Create a MediaPipe Hands object
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

while cap.isOpened():
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade classifier
    faces_haar = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces_haar:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes within the face region
        eyes_haar = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes_haar:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Detect faces for blink detection
    faces_blink = detector_blink(gray)

    for face in faces_blink:
        shape = predictor_blink(gray, face)
        shape = np.array([[point.x, point.y] for point in shape.parts()])

        left_eye = shape[36:42]
        right_eye = shape[42:48]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        if ear < 0.2:
            blink_counter += 1

        frame_counter += 1

    # Calculate blink rate
    if frame_counter > 1:
        blink_rate = blink_counter / frame_counter

    # Display blink rate on the frame
    blink_rate_display = "Blink Rate: {:.2f}".format(blink_rate)
    cv2.putText(frame, blink_rate_display, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Detect faces for gaze direction detection
    faces_gaze = detector_gaze(gray)

    for face in faces_gaze:
        shape = predictor_gaze(gray, face)
        shape = np.array([[point.x, point.y] for point in shape.parts()])

        left_eye = shape[42:48]
        right_eye = shape[36:42]

        left_eye_center = np.mean(left_eye, axis=0).astype(int)
        right_eye_center = np.mean(right_eye, axis=0).astype(int)

        gaze_direction = determine_gaze_direction(left_eye_center, right_eye_center)

        cv2.putText(frame, f'Gaze Direction: {gaze_direction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Detect hands using MediaPipe Hands
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_results = hands.process(rgb_frame)

    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            # Extract relevant hand landmarks
            hand_landmarks_list = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in hand_landmarks.landmark]
            
            # Check if hand landmarks intersect with any detected faces
            for (x, y, w, h) in faces_haar:
                # Check if any hand landmark is within the bounding box of the face
                for landmark_x, landmark_y in hand_landmarks_list:
                    if x < landmark_x < x + w and y < landmark_y < y + h:
                        cv2.putText(frame, "Hand covering face", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        break  # No need to check other landmarks for this face

    # Use the FER model to detect emotions in the frame
    emotions = fer_detector.detect_emotions(frame)

    if emotions:
        # Get the emotion with the highest confidence
        emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)

        # Display the emotion on the frame
        cv2.putText(frame, emotion, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame with detected faces, emotions, and additional annotations
    cv2.imshow('Combined Detection', frame)

    # Exit the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
