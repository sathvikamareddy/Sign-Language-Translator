<div align="center">
  
# 🤟 Real-Time Sign Language Translator

</div>

A real-time Sign Language Recognition system that detects hand gestures using a webcam and translates them into alphabets (A–Z) using Machine Learning.

---

## 📌 Project Overview

This project uses computer vision and deep learning to recognize hand gestures and convert them into readable text. It is designed to help bridge communication between hearing-impaired individuals and others.

---

## 🚀 Features

* 🎥 Real-time webcam detection
* ✋ Hand tracking using MediaPipe
* 🔤 Predicts ASL alphabets (A–Z)
* ⚡ Fast prediction with confidence score
* 📊 Displays FPS (Frames Per Second)

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* MediaPipe
* NumPy

---

## 📂 Project Structure

SignProject/
│── pros.py # Main application file
│── asl_model.h5 # Trained model (not included)
│── requirements.txt # Dependencies
│── README.md # Project documentation

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

git clone https://github.com/your-username/Sign-Language-Translator.git

cd Sign-Language-Translator

---

### 2️⃣ Create virtual environment (recommended)

python -m venv tf_env

tf_env\Scripts\activate

---

### 3️⃣ Install dependencies

pip install -r requirements.txt

---

## ▶️ How to Run

python pros.py

---

## 📦 Model File

⚠️ The trained model file (`asl_model.h5`) is not included due to size limitations.

👉 To run the project:

* Place your trained model file in the project folder
  OR
* Update the file path in the code

---

## 🧠 How It Works

1. Captures live video from webcam
2. Detects hand landmarks using MediaPipe
3. Converts landmarks into feature vector
4. Uses trained TensorFlow model to predict the sign
5. Displays result in real-time

---

## 📸 Output

* Shows detected hand landmarks
* Displays predicted alphabet
* Shows confidence score and FPS

---

## 🎯 Future Improvements

* Word and sentence formation
* Voice output for predictions
* Support for dynamic gestures
* Mobile application integration

---


## ⭐ Acknowledgements

* MediaPipe for hand tracking
* TensorFlow for machine learning
* OpenCV for computer vision

---

## 📌 Note

This project is developed for educational and demonstration purposes.
