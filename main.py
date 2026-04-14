import cv2
import time
import threading
import winsound
import numpy as np
import pyttsx3
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision

# Download the task model if not exists
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading MediaPipe model...")
    url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
    urllib.request.urlretrieve(url, model_path)
    print("Download complete.")

# Initialize Face Landmarker (New Tasks API)
base_options = mp_tasks_python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1)
landmarker = vision.FaceLandmarker.create_from_options(options)

# Load cascades
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not opening")
    exit()

closed_eyes_start = None
drowsy_time = 0.0
focus_time = 0.0
session_time = 0.0
last_frame_time = time.time()
last_alarm_time = 0
last_yawn_alarm_time = 0

def play_alarm():
    winsound.Beep(2500, 500)

def speak_warning():
    try:
        engine = pyttsx3.init()
        engine.say("Please be focused!")
        engine.runAndWait()
    except Exception as e:
        print("Audio error:", e)

def draw_hud(frame, score, focus, drowsy, status, color):
    # Add a semi-transparent dark overlay (HUD) at the top-left
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (350, 240), (0, 0, 0), -1)
    
    alpha = 0.6  # Transparency factor
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Texts
    cv2.putText(frame, f"STATUS: {status}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Driver Score: {score}%", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Focus Time: {int(focus)}s", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 255, 150), 2)
    cv2.putText(frame, f"Drowsy Time: {int(drowsy)}s", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 255), 2)
    
    # Border
    cv2.rectangle(frame, (20, 20), (350, 240), (255, 255, 255), 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]

    current_time = time.time()
    delta_time = current_time - last_frame_time
    last_frame_time = current_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run MediaPipe Face Landmarker for Yawn Detection using Tasks API
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = landmarker.detect(mp_image)
    is_yawning = False
    
    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            # Lips Landmarks: 13 (upper inner lip), 14 (lower inner lip)
            lip_top = face_landmarks[13]
            lip_bottom = face_landmarks[14]
            
            # Use relative distance to avoid camera scale issues
            dist = abs(lip_bottom.y - lip_top.y)
            
            # Draw landmarks to visualize yawn detection on lips
            top_y = int(lip_top.y * frame_h)
            bot_y = int(lip_bottom.y * frame_h)
            mid_x = int(lip_top.x * frame_w)
            
            cv2.circle(frame, (mid_x, top_y), 2, (0, 255, 255), -1)
            cv2.circle(frame, (mid_x, bot_y), 2, (0, 255, 255), -1)

            if dist > 0.05:  # threshold for open mouth/yawn
                is_yawning = True

    # Haar Cascade for Face & Eye drowsiness
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    status = "Awake 👀"
    color = (0, 255, 0)
    
    face_detected = len(faces) > 0
    if face_detected:
        session_time += delta_time
        
        # Take the largest face
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        (x, y, w, h) = faces[0]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

        # -----------------------
        # 🧍 Posture Detection
        # -----------------------
        if y > (frame_h * 0.45):  
            cv2.putText(frame, "Head Down! ⚠️", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            if current_time - last_alarm_time > 2.0:
                threading.Thread(target=play_alarm, daemon=True).start()
                last_alarm_time = current_time

        # -----------------------
        # 🥱 Yawn Logic
        # -----------------------
        if is_yawning:
            status = "Yawning 🥱"
            color = (0, 165, 255)
            cv2.putText(frame, "YAWN ALERT!", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if current_time - last_yawn_alarm_time > 5.0: # 5 sec cooldown
                threading.Thread(target=speak_warning, daemon=True).start()
                last_yawn_alarm_time = current_time

        # -----------------------
        # 👁️ Eye Logic inside ROI
        # -----------------------
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)

        if len(eyes) == 0:
            if closed_eyes_start is None:
                closed_eyes_start = current_time

            sleep_time = current_time - closed_eyes_start

            if sleep_time > 2.0:
                status = "Drowsy 🔴"
                color = (0, 0, 255)
                drowsy_time += delta_time
                
                if current_time - last_alarm_time > 1.0:
                    threading.Thread(target=play_alarm, daemon=True).start()
                    last_alarm_time = current_time

            elif sleep_time > 1.0:
                status = "Tired 🟡"
                color = (0, 255, 255)
                focus_time += delta_time
            else:
                if not is_yawning:
                    status = "Blinking..."
                focus_time += delta_time

        else:
            closed_eyes_start = None
            focus_time += delta_time
            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 2)
    else:
        status = "No Face ❓"
        color = (150, 150, 150)
        closed_eyes_start = None

    # -----------------------
    # 📊 Driver Score
    # -----------------------
    if session_time > 0:
        score = int((focus_time / session_time) * 100)
        score = min(100, max(0, score))
    else:
        score = 100

    # -----------------------
    # 🖥️ Display HUD
    # -----------------------
    draw_hud(frame, score, focus_time, drowsy_time, status, color)

    cv2.imshow("AI Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()