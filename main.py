import torch
from torchvision import transforms
import cv2

# Loading pretrained models using torch.hub.load()
model_detect_raised_eyebrows = torch.hub.load('MachinaDominus/srp', 'raised_eyebrows', pretrained=True)
model_detect_raised_eyebrows.eval()

model_detect_trembling_hands = torch.hub.load('MachinaDominus/srp', 'trembling_hands', pretrained=True)
model_detect_trembling_hands.eval()

model_detect_avoiding_eye_contact = torch.hub.load('MachinaDominus/srp', 'avoiding_eye_contact', pretrained=True)
model_detect_avoiding_eye_contact.eval()

model_detect_eyes_squinting = torch.hub.load('MachinaDominus/srp', 'eyes_squinting', pretrained=True)
model_detect_eyes_squinting.eval()

model_detect_lip_twiching = torch.hub.load('MachinaDominus/srp', 'lip_twitching', pretrained=True)
model_detect_lip_twiching.eval()

model_detect_crossing_arms = torch.hub.load('MachinaDominus/srp', 'crossing_arms', pretrained=True)
model_detect_crossing_arms.eval()

# Transformation for pre-processing images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Functions for behavior detection
def detect_raised_eyebrows(frame):
    input_tensor = transform(frame).unsqueeze(0)
    raised_eyebrows_detection_result = model_detect_raised_eyebrows(input_tensor)
    raised_eyebrows_detected = False
    return raised_eyebrows_detected

def detect_trembling_hands(frame):
    input_tensor = transform(frame).unsqueeze(0)
    trembling_hands_detection_result = model_detect_trembling_hands(input_tensor)
    trembling_hands_detected = False
    return trembling_hands_detected

def detect_avoiding_eye_contact(frame):
    input_tensor = transform(frame).unsqueeze(0)
    avoiding_eye_contact_detection_result = model_detect_avoiding_eye_contact(input_tensor)
    avoiding_eye_contact_detected = False
    return avoiding_eye_contact_detected

def detect_eyes_squinting(frame):
    input_tensor = transform(frame).unsqueeze(0)
    eyes_squinting_detection_result = model_detect_eyes_squinting(input_tensor)
    eyes_squinting_detected = False
    return eyes_squinting_detected

def detect_lip_twiching(frame):
    input_tensor = transform(frame).unsqueeze(0)
    lip_twiching_detection_result = model_detect_lip_twiching(input_tensor)
    lip_twiching_detected = False
    return lip_twiching_detected

def detect_crossing_arms(frame):
    input_tensor = transform(frame).unsqueeze(0)
    crossing_arms_detection_result = model_detect_crossing_arms(input_tensor)
    crossing_arms_detected = False
    return crossing_arms_detected

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    raised_eyebrows_detected = detect_raised_eyebrows(frame)
    trembling_hands_detected = detect_trembling_hands(frame)
    avoiding_eye_contact_detected = detect_avoiding_eye_contact(frame)
    eyes_squinting_detected = detect_eyes_squinting(frame)
    lip_twiching_detected = detect_lip_twiching(frame)
    crossing_arms_detected = detect_crossing_arms(frame)

    if raised_eyebrows_detected:
        cv2.putText(frame, "Raised Eyebrows Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if trembling_hands_detected:
        cv2.putText(frame, "Trembling Hands Detected", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if avoiding_eye_contact_detected:
        cv2.putText(frame, "Avoiding Eye Contact Detected", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if eyes_squinting_detected:
        cv2.putText(frame, "Eyes Squinting Detected", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if lip_twiching_detected:
        cv2.putText(frame, "Lip Twitching Detected", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if crossing_arms_detected:
        cv2.putText(frame, "Crossing Arms Detected", (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Behavior Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
