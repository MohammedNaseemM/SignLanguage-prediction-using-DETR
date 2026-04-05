# 🤟 Real-Time Sign Language Detection using DETR

A deep learning-based system that detects and classifies sign language gestures in real time using a webcam. The project leverages **Detection Transformers (DETR)** for accurate gesture localization and recognition.

---

## 🚀 Features

* 🎥 Real-time webcam-based detection
* 🧠 Transformer-based model (DETR)
* 📦 Bounding box localization + classification
* 🎯 Confidence-based filtering
* 📊 FPS and performance monitoring
* 🖥️ Modern UI using Streamlit
* 🔧 Fully modular and scalable pipeline

---

## 🧠 Tech Stack

* **Language:** Python 3.10
* **Deep Learning:** PyTorch, Torchvision
* **Computer Vision:** OpenCV
* **Model:** DETR (Detection Transformer)
* **Data Augmentation:** Albumentations
* **UI:** Streamlit
* **Annotation Tool:** Label Studio

---

## 📁 Project Structure

```
SignDETR-main/
│
├── app.py                 # Streamlit UI
├── src/
│   ├── realtime.py        # Real-time detection (CLI)
│   ├── train.py           # Model training
│   ├── data.py            # Dataset handling
│   ├── model.py           # DETR model
│   ├── utils/
│   │   ├── boxes.py
│   │   ├── logger.py
│   │   ├── setup.py
│   │   ├── rich_handlers.py
│
├── checkpoints/           # Trained model (.pt)
├── dataset/               # Images + annotations
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```
git clone <your-repo-link>
cd SignDETR-main
```

---

### 2️⃣ Create virtual environment

```
py -3.10 -m venv .venv
.venv\Scripts\activate
```

---

### 3️⃣ Install dependencies

```
python -m pip install "numpy<2"
python -m pip install opencv-python==4.9.0.80
python -m pip install torch torchvision
python -m pip install albumentations einops matplotlib rich torchinfo streamlit
```

---

## ▶️ Usage

### 🔹 Run Real-Time Detection (CLI)

```
python src/realtime.py
```

---

### 🔹 Run Streamlit UI

```
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 📸 Sample Output

> Add screenshots here for better presentation

* UI Interface
* Gesture Detection
* Bounding Box Output

---

## 🏗️ System Pipeline

```
Webcam → Preprocessing → DETR Model → Bounding Boxes → Display
```

---

## 📊 Model Details

* Architecture: DETR (Transformer-based object detection)
* Backbone: ResNet
* Input Size: 224 × 224
* Output:

  * Bounding boxes (x1, y1, x2, y2)
  * Class labels
  * Confidence scores

---

## 📈 Performance

* Real-time inference on CPU
* FPS: ~2–5 (depending on system)
* Accuracy improves with dataset size

---

## ⚠️ Known Issues

* Requires good lighting for better detection
* Lower FPS on CPU-only systems
* Limited gesture classes (can be expanded)

---

## 🔮 Future Improvements

* Multi-gesture recognition
* Sentence formation using NLP
* Mobile/Web deployment
* Model optimization for higher FPS

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is for educational and research purposes.

---

## 👨‍💻 Author

**Your Name**
BSc IT (Data Science)
[Your LinkedIn / GitHub]

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!

---
