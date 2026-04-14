# AI-Driven-Monitoring-System
Developed an AI-based driver monitoring system using OpenCV and MediaPipe to detect drowsiness, yawning, and attention levels in real-time, with alert mechanisms and performance tracking.

---

## 📌 Project Description

The AI Driver Monitoring System is designed to improve road safety by continuously monitoring the driver’s facial behavior using a webcam.

The system detects:
- 👁️ Eye closure (drowsiness)
- 🥱 Yawning detection
- 🧍 Head posture (head down)
- 😊 Face presence

It calculates a **Driver Score** based on focus time and alerts the user using sound and voice warnings when unsafe behavior is detected.

---

## ✨ Features

- 🎥 Real-time video processing using webcam  
- 👁️ Eye detection for drowsiness monitoring  
- 🥱 Yawn detection using facial landmarks  
- 🧍 Posture detection (head down alert)  
- 🔊 Audio alert system (beep + voice warning)  
- 📊 Driver Score calculation  
- ⏱️ Focus time & drowsy time tracking  
- 🖥️ Interactive HUD display  

---

## 🛠️ Technologies Used

- **Python**
- **OpenCV** – Video capture & image processing  
- **MediaPipe** – Facial landmark detection  
- **NumPy** – Mathematical calculations  
- **pyttsx3** – Voice alert system  
- **winsound** – Beep alert  

---

## ⚙️ How It Works

1. Captures live video using webcam  
2. Detects face using Haar Cascade  
3. Uses MediaPipe to detect facial landmarks  
4. Tracks:
   - Eye closure → drowsiness  
   - Lip distance → yawning  
   - Face position → posture  
5. Calculates:
   - Focus Time  
   - Drowsy Time  
   - Driver Score  
6. Triggers alerts when unsafe behavior is detected  

---

## 📂 Project Structure
AI-Driver-Monitoring-System/
│
├── main.py # Main application logic
├── requirements.txt # Dependencies
├── face_landmarker.task # MediaPipe model
└── README.md # Documentation


---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt

Requirements

From :

opencv-python
mediapipe
numpy
pyttsx3

Key Logic

From implementation :

Eye detection using Haar Cascade
Facial landmarks using MediaPipe
Yawning detection using lip distance
Drowsiness detection based on eye closure time
Driver score:
Driver Score = (Focus Time / Total Time) × 100

Alerts
🔊 Beep sound for drowsiness
🗣️ Voice alert: “Please be focused!”
⚠️ Visual warnings on screen
📊 Output

Displays:

Driver Status (Awake, Drowsy, Yawning, etc.)
Driver Score (%)
Focus Time
Drowsy Time

Author

Vedashri Giri
Computer Science Engineering Student

⭐ Acknowledgement

This project demonstrates the use of AI and Computer Vision in real-time safety applications such as driver monitoring systems.



