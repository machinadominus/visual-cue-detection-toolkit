import torch
from torchvision import transforms
import cv2

# loading my pretrained models from github using torch.hub.load()

model_detect_raised_eyebrows = torch.hub.load('MachinaDominus/srp', 'raised_eyebrows', pretrained = True)
model_detect_raised_eyebrows.eval()

model_detect_gaze_aversion = torch.hub.load('MachinaDominus/srp', 'gaze_aversion', pretrained = True)
model_detect_gaze_aversion.eval()

model_detect_sweating = torch.hub.load('MachinaDominus/srp', 'sweating' , pretrained = True)
model_detect_sweating.eval()

model_detect_trembling_hands = torch.hub.load('MachinaDominus/srp', 'trembling_hands' , pretrained = True)
model_detect_trembling_hands.eval()

model_detect_avoiding_eye_contact = torch.hub.load('MachinaDominus/srp', 'avoiding_eye_contact' , pretrained = True)
model_detect_avoiding_eye_contact.eval()

model_detect_tense_facial_muscles = torch.hub.load('MachinaDominus/srp', 'tense_facial_muscles' , pretrained = True)
model_detect_tense_facial_muscles.eval()

model_detect_eyes_squinting = torch.hub.load('MachinaDominus/srp', 'eyes_squinting' , pretrained = True)
model_detect_eyes_squinting.eval()

model_detect_lip_twiching = torch.hub.load('MachinaDominus/srp', 'lip_twitching' , pretrained = True)
model_detect_lip_twiching.eval()

model_detect_flaring_nostrils = torch.hub.load('MachinaDominus/srp', 'flaring_nostrils' , pretrained = True)
model_detect_flaring_nostrils.eval()

model_detect_crossing_arms = torch.hub.load('MachinaDominus/srp', 'crossing_arms' , pretrained = True)
model_detect_crossing_arms.eval()

# Transformation for pre-processing images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def detect_raised_eyebrows(frame):
  input_tensor = transform(frame).unsqueeze(0)
  raised_eyebrows_detection_result = model_detect_raised_eyebrows(input_tensor)

  raised_eyebrows_detected_ = False
  return raised_eyebrows_detected_

def detect_gaze_aversion(frame):
  input_tensor = transform(frame).unsqueeze(0)
  gaze_aversion_detection_result = model_detect_gaze_aversion(input_tensor)

  gaze_aversion_detected = False
  return gaze_aversion_detected

def detect_sweating(frame):
  input_tensor = transform(frame).unsqueeze(0)
  sweating_detection_result = model_detect_sweating(input_tensor)

  sweating_detected = False
  return sweating_detected

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

def detect_tense_facial_muscles(frame):
  input_tensor = transform(frame).unsqueeze(0)
  tense_facial_muscles_detection_result = model_detect_tense_facial_muscles(input_tensor)
  tense_facial_muscles_detected = False
  return tense_facial_muscles_detected

def detect_eyes_squinting(frame):
  input_tensor = transform(frame).unsqueeze(0)
  eyes_squinting_detection_result = model_detect_eyes_squinting(input_tensor)
  eyes_squinting_detected = False
  return eyes_squinting_detected

def detect_lip_twiching(frame):
  input_tensor = transform(frame).unsqueeze(0)
  eyes_squinting_detection_result = model_detect_eyes_squinting(input_tensor)
  lip_twiching_detected = False
  return lip_twiching_detected

def detect_flaring_nostrils(frame):
  input_tensor = transform(frame).unsqueeze(0)
  flaring_nostrils_detection_result = model_detect_flaring_nostrils(input_tensor)
  flaring_detected_detected = False
  return flaring_detected_detected

def detect_crossing_arms(frame):
  input_tensor = transform(frame).unsqueeze(0)
  crossing_arms_detection_result = model_detect_crossing_arms(input_tensor)
  crossing_arms_detected = False
  return crossing_arms_detected





# Capture video from webcam
cap = cv2.VideoCapture(0)

# ... (previous code)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect picking behavior in the current frame
    #picking_detected = detect_picking_behavior(frame)
    
    # Detect other behaviors here and assign the respective boolean variables
    raised_eyebrows_detected = detect_raised_eyebrows(frame)
    gaze_aversion_detected = detect_gaze_aversion(frame)
    sweating_detected = detect_sweating(frame)
    trembling_hands_detected = detect_trembling_hands(frame)
    avoiding_eye_contact_detected = detect_avoiding_eye_contact(frame)
    tense_facial_muscles_detected = detect_tense_facial_muscles(frame)
    eyes_squinting_detected = detect_eyes_squinting(frame)
    lip_twiching_detected = detect_lip_twiching(frame)
    flaring_nostrils_detected = detect_flaring_nostrils(frame)
    crossing_arms_detected = detect_crossing_arms(frame)

    # Display detected behaviors on the frame
    if picking_detected:
        cv2.putText(frame, "Picking Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if raised_eyebrows_detected:
        cv2.putText(frame, "Raised Eyebrows Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if gaze_aversion_detected:
        cv2.putText(frame, "Gaze Aversion Detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if sweating_detected:
        cv2.putText(frame, "Sweating Detected", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if trembling_hands_detected:
        cv2.putText(frame, "Trembling Hands Detected", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if avoiding_eye_contact_detected:
        cv2.putText(frame, "Avoiding Eye Contact Detected", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if tense_facial_muscles_detected:
        cv2.putText(frame, "Tense Facial Muscles Detected", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if eyes_squinting_detected:
        cv2.putText(frame, "Eyes Squinting Detected", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if lip_twiching_detected:
        cv2.putText(frame, "Lip Twiching Detected", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if flaring_nostrils_detected:
        cv2.putText(frame, "Flaring Nostrils Detected", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if crossing_arms_detected:
        cv2.putText(frame, "Crossing Arms Detected", (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Behavior Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

