import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib
from skimage.feature import hog
import pandas as pd

# ---------- SESSION ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------- LOGIN PAGE ----------
def login():
    st.set_page_config(page_title="Verifi.ai Login", layout="wide")

    # Background
    st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a;
    }
    .box {
        background: #ffffff;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1,1])

    # LEFT SIDE
    with col1:
        st.markdown("""
        <div style='padding:60px;color:white;'>
            <h2>Welcome to</h2>
            <h1>🧠 Verifi.ai</h1>
            <p>AI-powered system for detecting fake images using Machine Learning.</p>
        </div>
        """, unsafe_allow_html=True)

    # RIGHT SIDE
    with col2:
        st.markdown("<div class='box'>", unsafe_allow_html=True)

        st.markdown("### Login to your account")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Sign In", use_container_width=True):
            if username == "admin" and password == "1234":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials ❌")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------- SHOW LOGIN ----------
if not st.session_state.logged_in:
    login()
    st.stop()

# ---------- MAIN APP ----------
st.set_page_config(page_title="Verifi.ai", layout="wide")

# Sidebar logout
st.sidebar.title("Menu")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# Load model
model = joblib.load("model.pkl")

# ---------- UI ----------
st.markdown("""
<style>
.title {
    font-size: 45px;
    font-weight: bold;
}
.highlight {
    color: #27ae60;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    color: white;
    font-weight: bold;
    text-align: center;
}
.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    text-align: center;
    padding: 10px;
    background: #f1f1f1;
}
</style>
""", unsafe_allow_html=True)

# HERO
col1, col2 = st.columns([2,1])

with col1:
    st.markdown('<div class="title">Verifi.ai <span class="highlight">Fake Image Detection</span></div>', unsafe_allow_html=True)
    st.write("Upload an image to check whether it is real or fake")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

with col2:
    st.image("ver.png",
             use_container_width=True)

# DASHBOARD
st.markdown("## 📊 Dashboard")

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", "75%")
c2.metric("Images", "200")
c3.metric("Classes", "2")

data = pd.DataFrame({
    "Category": ["Real", "Fake"],
    "Count": [100, 100]
})
st.bar_chart(data.set_index("Category"))

# FEATURE
def extract_features(image):
    image = np.array(image)
    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    ).reshape(1, -1)

# PREDICTION
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)

    features = extract_features(image)
    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    conf = np.max(probs) * 100

    if conf < 70:
        result = "⚠️ Uncertain"
        color = "#f39c12"
    elif pred == 1:
        result = "❌ Fake"
        color = "#e74c3c"
    else:
        result = "✅ Real"
        color = "#2ecc71"

    st.markdown(f'<div class="result-box" style="background:{color};">{result}</div>', unsafe_allow_html=True)
    st.write(f"Confidence: {conf:.2f}%")

    chart = pd.DataFrame({
        "Label": ["Real", "Fake"],
        "Confidence": [probs[0]*100, probs[1]*100]
    })
    st.bar_chart(chart.set_index("Label"))

# FOOTER
st.markdown("""
<div style="
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: #020617;
    color: #38bdf8;
    text-align: center;
    padding: 12px;
    font-size: 14px;
    border-top: 1px solid #1e293b;
    z-index: 999;
">
🚀 Verifi.ai | Fake Image Detection System | Developed by AIML Students
</div>
""", unsafe_allow_html=True)