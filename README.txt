# 🖼️ Camera Classifier with Conv2D (Tkinter + TensorFlow)

This project was **inspired by Neural Nine's camera classifier**, but unlike his approach which used an **SVM (Support Vector Machine)** for classification, **this implementation uses a deep learning model (Conv2D CNN)** for much better accuracy and scalability.  

With this version, you can **collect images directly from your camera**, **train a CNN**, and **predict in real time** – all from a simple Tkinter GUI.

---

## 🚀 Features
- 📸 **Live camera feed** with frame capture buttons for two classes.
- 🗂️ Automatic dataset folder creation (`1/`, `2/`).
- 🧠 **Conv2D Neural Network** model for training instead of SVM.
- 🔄 Train button to re-train the model whenever you capture new data.
- 🔮 Predict button & **Auto Prediction mode** for real-time classification.
- 💾 Model is saved as `Camera_classifier.keras` and auto-loaded on startup.

---

## 📂 Project Structure

