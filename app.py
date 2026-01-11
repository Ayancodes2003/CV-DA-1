"""Streamlit app (restored)
"""
import io

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2

from shape_detector import ShapeDetector

st.set_page_config(page_title="Shape & Contour Analyzer", page_icon="🔷", layout="wide")
st.title("🔷 Shape & Contour Analyzer")
st.write("Interactive demo for contour-based shape detection using classical OpenCV methods.")

st.sidebar.header("Controls 🛠️")
canny1 = st.sidebar.slider("Canny Threshold 1", min_value=10, max_value=300, value=50)
canny2 = st.sidebar.slider("Canny Threshold 2", min_value=50, max_value=400, value=150)
min_area = st.sidebar.slider("Minimum Contour Area", min_value=10, max_value=5000, value=200)

st.sidebar.markdown("---")
st.sidebar.markdown("**Tips:** Upload clear images with simple non-overlapping shapes for best results.")

uploaded_file = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns([1, 1])

if uploaded_file is not None:
    image_data = uploaded_file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    sd = ShapeDetector()
    annotated, edges, detected = sd.detect_shapes(
        image_bgr, canny_thresh1=canny1, canny_thresh2=canny2, min_area=min_area
    )

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    edges_rgb = cv2.cvtColor(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)

    with st.expander("Preview images 👀", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.header("Original")
        c1.image(image, width=300)
        c2.header("Edges (Canny)")
        c2.image(edges_rgb, width=300)
        c3.header("Annotated")
        c3.image(annotated_rgb, width=300)

    st.markdown("---")
    left, right = st.columns([1, 2])
    object_count = len(detected)
    left.metric(label="Detected objects 🔢", value=object_count)

    if object_count > 0:
        df = pd.DataFrame(detected)
        df.index += 1
        right.subheader("Detected shapes")
        right.dataframe(df.rename_axis("#"))
        with st.expander("Technical details 🔧"):
            st.write("**Canny thresholds:**", canny1, ",", canny2)
            st.write("**Minimum contour area:**", min_area)
            st.write("**Detected items (sorted by area):**")
            st.table(df)
    else:
        right.info("No shapes detected. Try lowering the minimum area or adjusting Canny thresholds.")

    st.markdown("---")
    st.subheader("Export")
    annotated_pil = Image.fromarray(annotated_rgb)
    buf = io.BytesIO()
    annotated_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button("Download annotated image", byte_im, file_name="annotated.png", mime="image/png")

else:
    st.info("Upload an image to begin shape analysis. Examples: simple geometric shapes on plain backgrounds.")

st.markdown("---")
st.caption("Built with OpenCV & Streamlit — classical computer vision techniques only. Suitable for academic evaluation.")
