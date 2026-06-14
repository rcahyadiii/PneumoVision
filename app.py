"""
PneumoVision — Pneumonia Detection from Chest X-Rays
Streamlit web app for the Digital Image Processing final project.

Run:
    pip install -r requirements.txt
    streamlit run app.py

It expects the trained artifacts (from the Colab notebook) in ./artifacts :
    artifacts/densenet_pneumonia.keras
    artifacts/svm_pneumonia.joblib
    artifacts/results.json
Override the location with the PNEUMO_ARTIFACTS environment variable if needed.
"""

import os
import json
import numpy as np
import cv2
import streamlit as st
from PIL import Image

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
IMG_SIZE = 224
CLASSES = ["NORMAL", "PNEUMONIA"]
ART = os.environ.get("PNEUMO_ARTIFACTS", "artifacts")

st.set_page_config(page_title="PneumoVision", page_icon="🫁", layout="wide")

# A little styling for a clean, clinical look
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.big-title {font-size: 2.1rem; font-weight: 700; margin-bottom: 0;}
.subtle {color: #6b7280; font-size: 0.95rem;}
.card {border: 1px solid #e5e7eb; border-radius: 12px; padding: 1rem 1.2rem; background: #ffffff;}
.disclaimer {background:#fef3c7; border:1px solid #fcd34d; color:#92400e;
             border-radius:8px; padding:.5rem .8rem; font-size:.85rem;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Preprocessing + features  (identical to the training notebook)
# ----------------------------------------------------------------------------
def preprocess_gray(gray, denoise="median", enhance="clahe"):
    """Resize -> denoise -> contrast-enhance. uint8 grayscale in/out."""
    g = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    if denoise == "median":
        g = cv2.medianBlur(g, 3)
    elif denoise == "gaussian":
        g = cv2.GaussianBlur(g, (3, 3), 0)
    if enhance == "clahe":
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    elif enhance == "hist_eq":
        g = cv2.equalizeHist(g)
    elif enhance == "stretch":
        lo, hi = np.percentile(g, (2, 98))
        g = np.clip((g - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)
    return g


def to_rgb(gray_u8):
    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB)


def extract_features(gray_u8):
    """GLCM texture + LBP histogram + Hu moments -> 1D feature vector (37 dims)."""
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    feats = []
    q = (gray_u8 / 4).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=64, symmetric=True, normed=True)
    for prop in ["contrast", "homogeneity", "energy", "correlation", "dissimilarity"]:
        feats.extend(graycoprops(glcm, prop).ravel())
    P, R = 8, 1
    lbp = local_binary_pattern(gray_u8, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=P + 2, range=(0, P + 2), density=True)
    feats.extend(hist)
    hu = cv2.HuMoments(cv2.moments(gray_u8)).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    feats.extend(hu)
    return np.array(feats, dtype=np.float32)


# ----------------------------------------------------------------------------
# Model loading (cached so it only happens once)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_classical():
    import joblib
    return joblib.load(os.path.join(ART, "svm_pneumonia.joblib"))


@st.cache_resource(show_spinner=False)
def load_deep():
    import tensorflow as tf
    model = tf.keras.models.load_model(os.path.join(ART, "densenet_pneumonia.keras"))
    # Build Grad-CAM helpers: feature extractor (image -> conv maps) + head (conv -> prob)
    feature_extractor, head_model = None, None
    try:
        base = None
        for layer in model.layers:                  # last 4-D layer = DenseNet base output
            try:
                if len(layer.output.shape) == 4:
                    base = layer
            except Exception:
                pass
        feature_extractor = tf.keras.Model(base.input, base.output)
        hin = tf.keras.Input(base.output.shape[1:])
        x = hin
        for name in ["gap", "drop", "pred"]:
            x = model.get_layer(name)(x)
        head_model = tf.keras.Model(hin, x)
    except Exception:
        feature_extractor, head_model = None, None
    return model, feature_extractor, head_model


@st.cache_data(show_spinner=False)
def load_results():
    try:
        with open(os.path.join(ART, "results.json")) as f:
            return json.load(f)
    except Exception:
        return None


def deep_predict(model, rgb_u8):
    """Model already contains DenseNet preprocessing; feed raw [0..255] RGB."""
    x = rgb_u8.astype("float32")[None, ...]
    return float(model.predict(x, verbose=0).ravel()[0])


def grad_cam(feature_extractor, head_model, rgb_u8):
    import tensorflow as tf
    from tensorflow.keras.applications.densenet import preprocess_input
    x = preprocess_input(rgb_u8.astype("float32")[None, ...])
    with tf.GradientTape() as tape:
        conv = feature_extractor(x)
        tape.watch(conv)
        p = head_model(conv)
    grads = tape.gradient(p, conv)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heat = cv2.applyColorMap((cam * 255).astype("uint8"), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(rgb_u8, 0.6, heat, 0.4, 0)
    return overlay


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown('<p class="big-title">🫁 PneumoVision</p>', unsafe_allow_html=True)
st.markdown('<p class="subtle">Pneumonia detection from chest X-rays — classical texture analysis vs. deep transfer learning.</p>',
            unsafe_allow_html=True)
st.markdown('<div class="disclaimer">⚠️ Research / education project only. This is <b>not</b> a medical device and must not be used for real diagnosis.</div>',
            unsafe_allow_html=True)
st.write("")

# Guard: artifacts present?
if not os.path.isdir(ART):
    st.error(f"Could not find the artifacts folder at **{ART}/**.\n\n"
             "Unzip `pneumovision_artifacts.zip` (from the Colab notebook) so that "
             "`densenet_pneumonia.keras`, `svm_pneumonia.joblib`, and `results.json` "
             "sit inside a folder named `artifacts/` next to this app — "
             "or set the `PNEUMO_ARTIFACTS` environment variable to its path.")
    st.stop()

# ----------------------------------------------------------------------------
# Sidebar — model selector + metrics
# ----------------------------------------------------------------------------
results = load_results()
with st.sidebar:
    st.header("Settings")
    model_choice = st.radio(
        "Classifier",
        ["Deep learning (DenseNet121)", "Classical (SVM + texture features)"],
    )
    st.divider()
    st.subheader("Test-set performance")
    if results:
        def auc_of(k):
            try:
                return f"{results[k]['auc']:.3f}"
            except Exception:
                return "—"
        st.metric("DenseNet121 ROC-AUC", auc_of("cnn"))
        st.metric("SVM ROC-AUC", auc_of("svm"))
        st.metric("Random Forest ROC-AUC", auc_of("rf"))
    else:
        st.caption("results.json not found — metrics unavailable.")
    st.divider()
    st.caption("Pipeline: resize → median denoise → CLAHE → classify.")

# ----------------------------------------------------------------------------
# Main — upload & predict
# ----------------------------------------------------------------------------
tab_predict, tab_perf, tab_about = st.tabs(["🔍 Predict", "📊 Model performance", "ℹ️ About"])

with tab_predict:
    file = st.file_uploader("Upload a chest X-ray (JPEG / PNG)", type=["jpg", "jpeg", "png"])

    if file is None:
        st.info("Upload a frontal chest X-ray to get started.")
    else:
        # Read as grayscale
        pil = Image.open(file).convert("L")
        gray = np.array(pil)

        # ---- Preprocessing view ----
        st.subheader("1 · Preprocessing stages")
        orig = preprocess_gray(gray, "none", "none")
        den = preprocess_gray(gray, "median", "none")
        enh = preprocess_gray(gray, "median", "clahe")
        c1, c2, c3 = st.columns(3)
        c1.image(orig, caption="Original (resized)", use_container_width=True, clamp=True)
        c2.image(den, caption="Median denoised", use_container_width=True, clamp=True)
        c3.image(enh, caption="+ CLAHE enhanced", use_container_width=True, clamp=True)

        # ---- Prediction ----
        st.subheader("2 · Prediction")
        with st.spinner("Running the model..."):
            if model_choice.startswith("Deep"):
                try:
                    model, fe, hm = load_deep()
                    prob = deep_predict(model, to_rgb(enh))
                    used = "DenseNet121 (transfer learning)"
                    overlay = None
                    if fe is not None and hm is not None:
                        try:
                            overlay = grad_cam(fe, hm, to_rgb(enh))
                        except Exception:
                            overlay = None
                except Exception as e:
                    st.error(f"Could not load / run the deep model: {e}\n\n"
                             "Make sure TensorFlow is installed and the .keras file is present.")
                    st.stop()
            else:
                try:
                    svm = load_classical()
                    feats = extract_features(enh).reshape(1, -1)
                    prob = float(svm.predict_proba(feats)[0, 1])
                    used = "SVM on GLCM/LBP/Hu features"
                    overlay = None
                except Exception as e:
                    st.error(f"Could not load / run the classical model: {e}")
                    st.stop()

        label = CLASSES[int(prob >= 0.5)]
        confidence = prob if label == "PNEUMONIA" else 1 - prob

        left, right = st.columns([1, 1])
        with left:
            if label == "PNEUMONIA":
                st.error(f"### Prediction: PNEUMONIA")
            else:
                st.success(f"### Prediction: NORMAL")
            st.write(f"**Confidence:** {confidence*100:.1f}%")
            st.progress(min(max(confidence, 0.0), 1.0))
            st.caption(f"Model used: {used}  ·  P(pneumonia) = {prob:.3f}")
        with right:
            if overlay is not None:
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                         caption="Grad-CAM — regions that drove the decision",
                         use_container_width=True)
            elif model_choice.startswith("Deep"):
                st.caption("Grad-CAM unavailable for this model build.")
            else:
                st.caption("Grad-CAM applies to the deep model only "
                           "(the classical model uses handcrafted features, not spatial activations).")

with tab_perf:
    st.subheader("Held-out test-set results (624 images)")
    if results:
        rows = [
            ("SVM (texture features)", "svm"),
            ("Random Forest (texture features)", "rf"),
            ("DenseNet121 (transfer learning)", "cnn"),
        ]
        table = []
        for name, key in rows:
            auc = results.get(key, {}).get("auc", None)
            table.append({"Model": name, "ROC-AUC": f"{auc:.3f}" if auc is not None else "—"})
        st.table(table)
    st.markdown(
        "From the full project report: the deep model reached **85.3%** accuracy / "
        "**0.957** ROC-AUC, versus ~**76%** / **0.802–0.838** for the classical models. "
        "DenseNet121 has very high pneumonia recall (96.4%) — the safer error profile for a "
        "screening tool — but lower specificity (it over-calls some normal cases)."
    )

with tab_about:
    st.markdown(
        "**PneumoVision** compares two ways of detecting pneumonia in chest X-rays:\n\n"
        "1. **Classical** — median denoising + CLAHE, then GLCM texture, Local Binary "
        "Patterns, and Hu shape moments fed to an SVM.\n"
        "2. **Deep learning** — a DenseNet121 (the CheXNet backbone) fine-tuned with "
        "transfer learning on the same CLAHE-enhanced images.\n\n"
        "Dataset: Kermany et al. (2018), *Chest X-Ray Images (Pneumonia)*.\n\n"
        "_Digital Image Processing final project. Educational use only._"
    )
