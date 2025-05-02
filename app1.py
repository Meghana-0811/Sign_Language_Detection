import streamlit as st
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

# ------------------- Load the trained model -------------------
model = load_model("sign_language_model.h5")

# ------------------- Load class labels -------------------
def load_class_labels(file_path):
    with open(file_path, 'r') as f:
        classes = f.read().splitlines()
    return classes

class_labels = load_class_labels("wlasl_class_list.txt")

# ------------------- Preprocess input frame -------------------
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    return frame_expanded, frame_resized

# ------------------- Make prediction -------------------
def predict_sign_language(frame):
    processed_frame, display_frame = preprocess_frame(frame)
    predictions = model.predict(processed_frame)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]
    return class_labels[predicted_index], confidence, display_frame

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Sign Language Detection", layout="wide")

# Page header
st.markdown(
    """
    <style>
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #4B8BBE;
        }
        .sub {
            text-align: center;
            font-size: 18px;
            color: #333;
        }
    </style>
    <div class="title"> Real-Time Sign Language Detection</div>
    <div class="sub">Detects sign language from webcam input.</div>
    <br>
    """,
    unsafe_allow_html=True
)

# Sidebar controls
st.sidebar.header("Control Panel")
run = st.sidebar.checkbox(" Start Webcam")
show_processed = st.sidebar.checkbox(" Show Preprocessed Frame")

# Main layout: webcam on left, prediction results on right
col1, col2 = st.columns([2, 1])
with col1:
    frame_display = st.empty()
    if show_processed:
        processed_display = st.empty()
with col2:
    result_display = st.empty()
    fps_display = st.empty()

if run:
    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error(" Could not access the webcam. Make sure it is connected!")
            break

        # Prediction
        try:
            prediction, confidence, processed_frame = predict_sign_language(frame)
            result_display.markdown(f"""
                <div style="background-color:#e0f7fa;padding:20px;border-radius:10px;">
                    <h2 style="color:#00796b;"> Detected Sign:</h2>
                    <h1 style="color:#004d40;">{prediction}</h1>
                    <p style="font-size:18px;">Confidence: <b>{confidence * 100:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            result_display.error(f"Prediction Error: {e}")
            continue

        # Show webcam frame
        frame_display.image(frame, channels="BGR", caption="📷 Live Webcam Feed")

        # Optionally show preprocessed input
        if show_processed:
            processed_display.image(processed_frame, channels="BGR", caption=" Preprocessed Frame")

        # Show FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_display.markdown(f"<p style='font-size:16px;'> <b>FPS:</b> {fps:.2f}</p>", unsafe_allow_html=True)

    cap.release()
else:
    st.info(" Check the 'Start Webcam' box in the sidebar to begin.")
