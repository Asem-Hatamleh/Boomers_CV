import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load YOLO models sdasd
model1 = YOLO("best_8n.pt")
model2 = YOLO("best_11n.pt")

st.title("🔪 Knife Detection (YOLO – Two Model Comparison)")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert file to image
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.write("Running detections... ⏳")

    # Predict using both models
    results1 = model1(img)
    results2 = model2(img)

    img_out = img.copy()

    # Draw boxes from model 1 (RED)
    for box in results1[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img_out, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_out, "Model 8n", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Draw boxes from model 2 (BLUE)
    for box in results2[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(img_out, "Model 11n", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    st.image(img_out, caption="Detections", use_column_width=True)


