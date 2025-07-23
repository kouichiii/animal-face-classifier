import pathlib
import pickle

import numpy as np
import streamlit as st
from PIL import Image
from rembg import remove

ASSETS = pathlib.Path(__file__).parent

@st.cache_resource
def load_models():
    with open(ASSETS/"pca.pkl", "rb") as f:
        pca = pickle.load(f)
    with open(ASSETS/"svm.pkl", "rb") as f:
        clf = pickle.load(f)
    return pca, clf

pca, clf = load_models()

st.set_page_config(
    page_title="Cat vs Dog Face Classifier",
    page_icon="🐱😺",
    layout="centered"
)

st.title("🐱😺 Cat vs Dog Face Classifier")
st.write("Upload a face image and this app will tell you how **cat-like** or **dog-like** it is.")

uploaded = st.file_uploader(
    "Choose an image file (JPEG/PNG)",
    type=["jpg", "jpeg", "png"]
)

def preprocess(img: Image.Image, size=(128, 128)) -> np.ndarray:
    # Background removal
    fg = remove(img)

    # Paste onto white background
    bg = Image.new("RGB", fg.size, (255, 255, 255))
    if fg.mode == "RGBA":
        bg.paste(fg, mask=fg.split()[-1])
    else:
        bg = fg.convert("RGB")

    # Grayscale then back to 3-channel
    gray = bg.convert("L")
    rgb_gray = Image.merge("RGB", (gray, gray, gray))

    # Resize and flatten
    arr = np.array(rgb_gray.resize(size), dtype=np.uint8)
    return arr.reshape(1, -1).astype(np.float32)


if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_column_width=True)

    X_pca = pca.transform(preprocess(img))
    prob  = clf.predict_proba(X_pca)[0]
    pred  = int(np.argmax(prob))

    animal = "Cat-like 🐱" if pred == 0 else "Dog-like 🐶"
    st.subheader(f"Prediction: **{animal}**")

    # Display probabilities
    st.metric("Cat probability", f"{prob[0]:.3f}")
    st.metric("Dog probability", f"{prob[1]:.3f}")
