# MindTune

## Quantum-Enhanced Emotion Detection for AI-Generated Music Therapy

MindTune is an intelligent multimodal system designed to detect a user’s emotional state in real time and generate adaptive audio responses such as calming ambient sounds or AI-generated music.

The project combines physiological sensing, facial emotion recognition, machine learning, quantum-enhanced classification research, and generative AI into one unified prototype.

---

# Project Objectives

* Detect emotional states such as calm and stress in real time
* Use wearable and sensor-based physiological signals
* Integrate facial emotion recognition through webcam input
* Compare classical and quantum machine learning approaches
* Generate adaptive music or ambient sound responses
* Provide an interactive real-time user interface

---

# Main Features

## Real-Time Physiological Sensing

Supports:

* Bluetooth HRM belt (Heart Rate + RR intervals)
* Temperature sensor
* Motion sensor (accelerometer / gyroscope)

## Facial Emotion Recognition

Uses:

* OpenCV
* DeepFace

Detects emotions through webcam input and confidence scores.

## Machine Learning Classification

Classical models tested:

* Random Forest
* Logistic Regression
* Support Vector Machine (SVM)

Quantum model tested:

* Quantum Support Vector Machine (QSVM)

## Adaptive Audio Response

Two response modes:

* AI-generated music using MusicGen
* Ambient relaxing sounds

## User Interface

Built with:

* Streamlit

---

# Final Architecture

Sensors + Webcam
↓
Feature Extraction
↓
Emotion Classification
↓
Emotion Fusion
↓
Adaptive Audio Generation
↓
Streamlit Interface

---

# Technologies Used

## Programming

* Python

## AI / ML

* Scikit-learn
* DeepFace
* TensorFlow
* Qiskit

## Interface

* Streamlit

## Audio

* MusicGen

## Hardware

* Arduino
* HRM Belt
* MPU6050
* DS18B20

---

# Project Structure

```text
MindTune/
│── interface/
│   └── app.py
│── emotion_detection/
│   ├── facial_emotion.py
│   └── emotion_fusion.py
│── audio/
│   └── audio_generator.py
│── models/
│   ├── classical/
│   └── quantum/
│── data/
│── results/
│── report_figures/
│── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/MindTune.git
cd MindTune
```

## Create Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

## Install Requirements

```bash
pip install -r requirements.txt
```

---

# Run Application

```bash
streamlit run interface/app.py
```

---

# Experimental Findings

* Random Forest provided the best real-time deployment performance.
* QSVM achieved competitive results under selected configurations.
* MusicGen improved realism compared with earlier MIDI pipelines.
* Multimodal sensing improved reliability over single-source detection.

---

# Limitations

* Limited custom dataset size
* MusicGen generation latency
* EDA not available in final real-time setup

---

# Future Work

* Larger physiological dataset collection
* Faster MusicGen generation
* Voice emotion recognition
* Multi-class emotion classification
* Advanced quantum models beyond QSVM

---

# Research Contribution

MindTune demonstrates how classical AI, quantum machine learning, physiological sensing, and generative media can be integrated into a single real-time human-centered system.

---

# Author

Kenza Qribis
Al Akhawayn University
Senior Capstone Project


