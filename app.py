import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import sys
import cv2
import numpy as np
import gdown

st.title("🥗 Vegetable Detection App (YOLO)")

# ------------------ Download models if not exist ------------------
MODELS = {
    "YOLOv8 Model": ("final_v8s.pt", "https://drive.google.com/uc?id=1rxT3IlsI6su5hKidPO3cX5QO1Jc77BUb"),
    "YOLOv11 Model": ("final_11s.pt", "https://drive.google.com/uc?id=1VIbH0YrC89XGSIF3imoghsMi-EkupjQJ"),
    "old 11s": ("best11s.pt", "https://drive.google.com/uc?id=1XyQoYHQTxqz5xne1MEbVxAEIvNJF-mvu"),
    "old v8s": ("bestv8s.pt", "https://drive.google.com/uc?id=1KGREDDEfEdFYNWsTxu3fw0QRfwYEqBi4")
    
}

os.makedirs("models", exist_ok=True)
for model_name, (filename, gdrive_url) in MODELS.items():
    model_path = os.path.join("models", filename)
    if not os.path.exists(model_path):
        st.info(f"Downloading {filename}...")
        gdown.download(gdrive_url, model_path, quiet=False)

# ------------------ Model selection ------------------
model_choice = st.selectbox("Select YOLO model", list(MODELS.keys()))
MODEL_PATH = os.path.join("models", MODELS[model_choice][0])

# ------------------ File Upload ------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    MIN_CONF_THRESHOLD = 0.5
    DISPLAY_RESOLUTION = '640x640'
    SAVE_ANNOTATED_IMAGE = True
    OUTPUT_IMAGE_PATH = os.path.join("./annotated_images", f"annotated_{uploaded_file.name}")
    os.makedirs("annotated_images", exist_ok=True)

    try:
        model = YOLO(MODEL_PATH, task='detect')
        labels = model.names
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        sys.exit(0)

    try:
        resW, resH = map(int, DISPLAY_RESOLUTION.split('x'))
        resized_frame = cv2.resize(frame, (resW, resH))
    except ValueError:
        st.error("Invalid display resolution format. Use 'WxH'.")
        sys.exit(0)

    results = model(resized_frame, verbose=False)
    detections = results[0].boxes

    object_count = 0
    annotated_frame = resized_frame.copy()
    bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
                   (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > MIN_CONF_THRESHOLD:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(annotated_frame, (xmin,ymin), (xmax,ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(annotated_frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(annotated_frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            object_count += 1

    cv2.putText(annotated_frame, f'Number of objects: {object_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, caption="Detected Vegetables", use_container_width=True)

    if SAVE_ANNOTATED_IMAGE:
        try:
            cv2.imwrite(OUTPUT_IMAGE_PATH, annotated_frame)
            st.success(f"Annotated image saved as '{OUTPUT_IMAGE_PATH}'")
        except Exception as e:
            st.error(f"Error saving the annotated image: {e}")
