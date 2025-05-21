# Drowsiness Detection Using Facial Landmarks

This Python project detects drowsiness in real-time using a webcam by monitoring eye aspect ratio (EAR) and mouth aspect ratio (MAR) using facial landmarks.

---

## 🎯 Features

- Real-time detection using OpenCV
- Eye aspect ratio (EAR) to detect eye closure
- Mouth aspect ratio (MAR) to detect yawning
- Alerts the user with a warning if drowsiness is detected

---

## 🧠 How It Works

The project uses `dlib`'s 68-point facial landmark predictor to locate key points around the eyes and mouth. EAR and MAR are calculated and used to classify whether the subject is drowsy.

---
