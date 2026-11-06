
import os
import time
import io

import cv2
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from PIL import Image

from src.classical import enhance_classical
from src.deep_learning import (
    load_zero_dce_model,
    enhance_zero_dce
)
from src.quality_metrics import calculate_quality_metrics


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Low-Light Image Enhancement",
    page_icon="🌙",
    layout="wide"
)

st.title("🌙 Low-Light Image Enhancement")

st.write(
    "Enhance low-light images using Classical Computer Vision, "
    "Zero-DCE Deep Learning and Hybrid Image Enhancement."
)


# ==========================================
# 2. LOAD ZERO-DCE MODEL
# ==========================================

@st.cache_resource
def get_zero_dce_model():

    model_path = "models/Epoch99.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    return load_zero_dce_model(model_path)


# ==========================================
# 3. HYBRID ENHANCEMENT
# ==========================================

def enhance_hybrid(classical_image, deep_image):

    start_time = time.perf_counter()

    if classical_image.shape != deep_image.shape:
        raise ValueError(
            "Classical and Zero-DCE images must have "
            "the same dimensions."
        )

    # Combine 85% Classical CV and 15% Zero-DCE
    hybrid = cv2.addWeighted(
        classical_image, 0.85,
        deep_image, 0.15,
        0
    )

    # Apply mild sharpening
    blurred = cv2.GaussianBlur(
        hybrid,
        (0, 0),
        1.0
    )

    hybrid = cv2.addWeighted(
        hybrid, 1.15,
        blurred, -0.15,
        0
    )

    hybrid = np.clip(
        hybrid, 0, 255
    ).astype(np.uint8)

    fusion_time = time.perf_counter() - start_time

    return hybrid, fusion_time


# ==========================================
# 4. CONVERT IMAGE TO PNG
# ==========================================

def convert_to_png_bytes(image):

    buffer = io.BytesIO()

    Image.fromarray(image).save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()


# ==========================================
# 5. CALCULATE QUALITY METRICS
# ==========================================

def get_quality_table(images):

    rows = []

    for method_name, image in images.items():

        metrics = calculate_quality_metrics(image)

        rows.append({
            "Method": method_name,
            "Brightness": metrics["Brightness"],
            "Contrast": metrics["Contrast"],
            "Sharpness": metrics["Sharpness"]
        })

    return pd.DataFrame(rows)


# ==========================================
# 6. IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a low-light image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is None:

    st.info(
        "Please upload a low-light image to start enhancement."
    )

else:

    try:

        pil_image = Image.open(uploaded_file).convert("RGB")
        original = np.array(pil_image)

        st.subheader("Original Image")

        preview_left, preview_center, preview_right = st.columns(
            [1, 2, 1]
        )

        with preview_center:

            st.image(
                original,
                caption="Uploaded Image",
                width=450
            )

        st.caption(
            f"Image dimensions: "
            f"{original.shape[1]} × {original.shape[0]}"
        )

        # ==========================================
        # 7. ENHANCE IMAGE
        # ==========================================

        if st.button(
            "Enhance Image",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Enhancing image. Please wait..."
                ):

                    total_start = time.perf_counter()

                    # Load trained Zero-DCE model
                    model = get_zero_dce_model()

                    # Classical CV enhancement
                    # Illumination output is not needed
                    classical_image, _, classical_time = (
                        enhance_classical(original)
                    )

                    # Zero-DCE deep learning enhancement
                    deep_image, deep_time = enhance_zero_dce(
                        original,
                        model
                    )

                    # Hybrid enhancement
                    hybrid_image, fusion_time = enhance_hybrid(
                        classical_image,
                        deep_image
                    )

                    total_processing_time = (
                        time.perf_counter() - total_start
                    )

                    # Hybrid end-to-end processing time
                    hybrid_total_time = (
                        classical_time +
                        deep_time +
                        fusion_time
                    )

                st.success(
                    "All enhancement methods completed successfully!"
                )

                # ==========================================
                # 8. ENHANCEMENT COMPARISON
                # ==========================================

                st.divider()
                st.header("Enhancement Comparison")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.image(
                        original,
                        caption="Original Image",
                        width="stretch"
                    )

                with col2:
                    st.image(
                        classical_image,
                        caption="Classical CV",
                        width="stretch"
                    )

                with col3:
                    st.image(
                        deep_image,
                        caption="Zero-DCE Deep Learning",
                        width="stretch"
                    )

                with col4:
                    st.image(
                        hybrid_image,
                        caption="Hybrid Enhancement",
                        width="stretch"
                    )

                # ==========================================
                # 9. PROCESSING TIME COMPARISON
                # ==========================================

                st.divider()
                st.header("Processing Time Comparison")

                time_col1, time_col2, time_col3 = st.columns(3)

                with time_col1:
                    st.metric(
                        "Classical CV",
                        f"{classical_time:.4f} sec"
                    )

                with time_col2:
                    st.metric(
                        "Zero-DCE",
                        f"{deep_time:.4f} sec"
                    )

                with time_col3:
                    st.metric(
                        "Hybrid End-to-End",
                        f"{hybrid_total_time:.4f} sec"
                    )

                st.caption(
                    "Hybrid End-to-End includes Classical CV "
                    "processing, Zero-DCE inference and final "
                    "image fusion."
                )

                st.caption(
                    f"Hybrid fusion and sharpening: "
                    f"{fusion_time:.6f} sec"
                )

                st.caption(
                    f"Total application processing time: "
                    f"{total_processing_time:.4f} sec"
                )

                # ==========================================
                # 10. IMAGE QUALITY METRICS
                # ==========================================

                st.divider()
                st.header("Image Quality Metrics")

                st.write(
                    "Comparison of brightness, contrast and "
                    "sharpness before and after enhancement."
                )

                images = {
                    "Original Image": original,
                    "Classical CV": classical_image,
                    "Zero-DCE": deep_image,
                    "Hybrid Enhancement": hybrid_image
                }

                quality_df = get_quality_table(images)

                method_order = [
                    "Original Image",
                    "Classical CV",
                    "Zero-DCE",
                    "Hybrid Enhancement"
                ]

                quality_df["Method"] = pd.Categorical(
                    quality_df["Method"],
                    categories=method_order,
                    ordered=True
                )

                quality_df = (
                    quality_df
                    .sort_values("Method")
                    .reset_index(drop=True)
                )

                quality_df["Method"] = (
                    quality_df["Method"].astype(str)
                )

                st.dataframe(
                    quality_df,
                    use_container_width=True,
                    hide_index=True
                )

                # ==========================================
                # 11. QUALITY METRICS VISUALIZATION
                # ==========================================

                st.divider()
                st.subheader("Quality Metrics Visualization")

                chart_df = quality_df.copy()

                chart_df["Method"] = pd.Categorical(
                    chart_df["Method"],
                    categories=method_order,
                    ordered=True
                )

                chart_df = chart_df.sort_values("Method")

                def create_quality_chart(metric_name):

                    chart = (
                        alt.Chart(chart_df)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "Method:N",
                                title="Method",
                                sort=method_order,
                                axis=alt.Axis(
                                    labelAngle=0,
                                    labelLimit=250
                                )
                            ),
                            y=alt.Y(
                                f"{metric_name}:Q",
                                title=metric_name,
                                scale=alt.Scale(zero=True)
                            ),
                            tooltip=[
                                alt.Tooltip(
                                    "Method:N",
                                    title="Method"
                                ),
                                alt.Tooltip(
                                    f"{metric_name}:Q",
                                    title=metric_name,
                                    format=".2f"
                                )
                            ]
                        )
                        .properties(height=400)
                    )

                    return chart

                metric1, metric2, metric3 = st.tabs([
                    "Brightness",
                    "Contrast",
                    "Sharpness"
                ])

                with metric1:

                    st.write("Brightness comparison")

                    st.altair_chart(
                        create_quality_chart("Brightness"),
                        use_container_width=True
                    )

                with metric2:

                    st.write("Contrast comparison")

                    st.altair_chart(
                        create_quality_chart("Contrast"),
                        use_container_width=True
                    )

                with metric3:

                    st.write("Sharpness comparison")

                    st.altair_chart(
                        create_quality_chart("Sharpness"),
                        use_container_width=True
                    )

                st.caption(
                    "Brightness measures average intensity, contrast "
                    "measures intensity variation, and sharpness is "
                    "estimated using Laplacian variance. Higher values "
                    "do not automatically mean better visual quality."
                )

                # ==========================================
                # 12. DOWNLOAD ENHANCED IMAGES
                # ==========================================

                st.divider()
                st.header("Download Enhanced Images")

                download_col1, download_col2, download_col3 = (
                    st.columns(3)
                )

                with download_col1:

                    st.download_button(
                        label="Download Classical CV",
                        data=convert_to_png_bytes(classical_image),
                        file_name="classical_enhanced.png",
                        mime="image/png",
                        use_container_width=True
                    )

                with download_col2:

                    st.download_button(
                        label="Download Zero-DCE",
                        data=convert_to_png_bytes(deep_image),
                        file_name="zero_dce_enhanced.png",
                        mime="image/png",
                        use_container_width=True
                    )

                with download_col3:

                    st.download_button(
                        label="Download Hybrid",
                        data=convert_to_png_bytes(hybrid_image),
                        file_name="hybrid_enhanced.png",
                        mime="image/png",
                        use_container_width=True
                    )

            except Exception as error:

                st.error(f"Enhancement failed: {error}")
                st.exception(error)

    except Exception as error:

        st.error(f"Unable to read image: {error}")
        st.exception(error)
