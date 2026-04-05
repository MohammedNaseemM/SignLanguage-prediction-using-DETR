import sys
import os
sys.path.append(os.path.abspath("src"))

import streamlit as st
import cv2
import torch
import time

from src.model import DETR
from src.utils.boxes import rescale_bboxes
from src.utils.setup import get_classes, get_colors
import albumentations as A

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sign Language Detection",
    layout="wide"
)

# ---------------- STYLING ----------------
st.markdown("""
    <style>
    .main-title {
        font-size:40px;
        font-weight:bold;
        text-align:center;
        color:#4CAF50;
    }
    .sub-text {
        text-align:center;
        color:gray;
        font-size:18px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🤟 Real-Time Sign Language Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">AI-powered gesture recognition using DETR</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")

run = st.sidebar.toggle("Start Camera", value=False)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.8)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    model = DETR(num_classes=3)
    model.eval()
    model.load_pretrained('checkpoints/99_model.pt')
    return model

model = load_model()

CLASSES = get_classes()
COLORS = get_colors()

# ---------------- TRANSFORM ----------------
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    A.pytorch.transforms.ToTensorV2()
])

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([3,1])  # Bigger camera area

FRAME_WINDOW = col1.image([])

status_box = col2.empty()
fps_box = col2.empty()

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

frame_count = 0
start_time = time.time()

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not working")
        break

    transformed = transform(image=frame)
    result = model(torch.unsqueeze(transformed['image'], dim=0))

    probs = result['pred_logits'].softmax(-1)[:,:,:-1]
    max_probs, max_classes = probs.max(-1)

    keep = max_probs > confidence_threshold

    h, w, _ = frame.shape
    bboxes = rescale_bboxes(result['pred_boxes'][keep], (w, h))
    classes = max_classes[keep]
    probas = max_probs[keep]

    for cls, prob, box in zip(classes, probas, bboxes):
        x1,y1,x2,y2 = box.detach().numpy()

        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),3)
        cv2.putText(frame,
                    f"{CLASSES[cls]} {prob:.2f}",
                    (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

    # ---------------- FPS ----------------
    frame_count += 1
    if frame_count % 10 == 0:
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        fps_box.metric("FPS", f"{fps:.2f}")

    # ---------------- STATUS ----------------
    if len(classes) > 0:
        status_box.success(f"Detected: {CLASSES[int(classes[0])]}")
    else:
        status_box.warning("No gesture detected")

    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()