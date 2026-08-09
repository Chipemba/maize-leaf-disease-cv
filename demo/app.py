"""
app.py
Streamlit used to live demo for ONNX maize leaf disease prediction.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from onnx_predict import predict_image


ONNX_MODEL_PATH = Path("onnx/mobilenet_v2.onnx")
DEFAULT_THRESHOLD = 0.5


st.set_page_config(
    page_title="Maize Leaf Disease ONNX Demo",
    page_icon="🌽",
    layout="wide"
)

st.title("Maize Leaf Disease Detection Demo")
st.caption("MobileNetV2 ONNX live inference for multi-label maize leaf disease prediction.")

st.markdown(
    """
This demo uses the trained **MobileNetV2** model exported to **ONNX**.  
The model predicts eight independent disease/condition probabilities:

`GLS`, `NCLB`, `PLS`, `CR`, `SR`, `NoFoliarSymptoms`, `Other`, and `UnidentifiedDisease`.

External images are used for qualitative live inference only. Predictions should be interpreted cautiously unless the external dataset labels fully match the original training labels.
"""
)

if not ONNX_MODEL_PATH.exists():
    st.error(
        f"ONNX model not found at `{ONNX_MODEL_PATH}`. "
        "Run `python src/deployment/export_onnx.py` first."
    )
    st.stop()

threshold = st.slider(
    "Prediction threshold",
    min_value=0.1,
    max_value=0.9,
    value=DEFAULT_THRESHOLD,
    step=0.05
)

uploaded_file = st.file_uploader(
    "Upload a maize leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("Uploaded image")
        st.image(image, use_container_width=True)

    with right_col:
        st.subheader("Predictions")

        results = predict_image(
            image=image,
            onnx_model_path=ONNX_MODEL_PATH,
            threshold=threshold
        )

        predicted_labels = [
            item for item in results if item["predicted"]
        ]

        if predicted_labels:
            st.success("Predicted labels above threshold:")
            for item in predicted_labels:
                st.write(
                    f"**{item['label']}** — {item['name']}: "
                    f"{item['probability']:.3f}"
                )
        else:
            st.warning(
                "No label passed the selected threshold. "
                "Try lowering the threshold or interpret the top probabilities cautiously."
            )

        results_df = pd.DataFrame(results)
        results_df["probability"] = results_df["probability"].round(4)

        st.subheader("All label probabilities")
        st.dataframe(
            results_df[["label", "name", "probability", "predicted"]],
            use_container_width=True
        )

        st.bar_chart(
            results_df.set_index("label")["probability"]
        )

else:
    st.info("Upload a maize leaf image to run live prediction.")