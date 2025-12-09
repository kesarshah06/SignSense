
# **SignSense — Real-Time ASL Gesture Recognition 🖐️🤖**

SignSense is a lightweight, modular pipeline for turning webcam hand gestures into real-time ASL predictions.
It uses MediaPipe Hands for landmark extraction and a simple but effective ML model for gesture classification.

This project lets you:

* 📸 Collect your own hand gesture dataset
* ✨ Extract 3D MediaPipe landmarks
* 🧠 Train a gesture classifier
* 🔍 Run real-time ASL prediction using your webcam

Clean, fast, and easy to extend.

---

## **📁 Repository Contents**

```
SignSense/
├── collect_data.py         # Collect images from webcam
├── extract_landmark.py     # Convert images → 63 MediaPipe features
├── training.py             # Train your ML model (MLP baseline)
├── train_model.ipynb       # Notebook version of full pipeline
├── detect_live.py          # Real-time prediction script
├── requirements.txt
├── setup.sh
└── gesture_baseline.pkl    # (Optional) Saved trained model
```

---

## **🚀 Getting Started**

### **1. Create and activate environment**

```bash
./setup.sh
source venv/bin/activate
```

Or manually:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## **📸 Step 1: Collect Training Images**

Run:

```bash
python collect_data.py
```

This script:

* Opens your webcam
* Cycles through gesture labels
* Saves images into automatically created folders

> Tip: Keep lighting consistent and your hand centered.

---

## **✨ Step 2: Extract Landmarks**

Convert all collected images into a single dataset:

```bash
python extract_landmark.py
```

This generates a CSV similar to:

```
x0, x1, x2, ..., x62, label
0.31, 0.52, 0.00, ..., A
...
```

This file is used for training (but **should not be uploaded to GitHub**).

---

## **🧠 Step 3: Train the Model**

```bash
python training.py
```

The script:

* Loads your landmarks file
* Splits train/test sets
* Scales features
* Trains an MLPClassifier
* Saves:

```
gesture_baseline.pkl
```

This file contains:

* model
* scaler
* label encoder

---

## **🔍 Step 4: Real-Time Detection**

```bash
python detect_live.py
```

This opens webcam detection and overlays:

* hand skeleton
* bounding box
* predicted ASL gesture

All in real time.

---

## **🎯 Future Improvements**

There are many ways to extend and enhance SignSense. Some recommended next steps:

### **1. Add More ASL Signs**

Expand the gesture vocabulary by collecting and labeling additional hand signs.
More classes → more expressive communication.

### **2. Increase Dataset Size**

Collect more training images per gesture in different:

* lighting conditions
* camera angles
* distances
* backgrounds

A larger, more diverse dataset greatly improves model accuracy and generalization.

### **3. Improve Model Stability**

Implement:

* temporal smoothing (EMA, rolling window majority vote)
* confidence thresholds
* prediction debounce

to reduce flicker in real-time detection.

### **4. Explore Better Models**

Try replacing the baseline MLP with:

* a small CNN trained on cropped hand images
* an LSTM/1D-CNN for dynamic gestures
* a transformer model on sequences of landmarks

### **5. Build a Simple UI**

Create a user interface using:

* Streamlit
* FastAPI + WebRTC
* Flask

to make the demo shareable and more interactive.

### **6. Add a Dataset Creation Wizard**

Automatically guide users through:

* camera calibration
* hand position guidance
* auto-sorting or cleaning images
* live preview during image capture

---


