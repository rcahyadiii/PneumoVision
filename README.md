# PneumoVision — Streamlit App

Pneumonia detection from chest X-rays, comparing a **classical** image-processing
pipeline (median + CLAHE → GLCM/LBP/Hu features → SVM) against a **deep**
transfer-learning model (DenseNet121). Final project for Digital Image Processing.

> ⚠️ Educational use only. This is **not** a medical device.

## 1. Get the trained models

These come from the training notebook (`PneumoVision_Training.ipynb`). After running it,
download `pneumovision_artifacts.zip` and unzip it so the folder looks like:

```
app.py
requirements.txt
artifacts/
├── densenet_pneumonia.keras
├── svm_pneumonia.joblib
└── results.json
```

If you keep the artifacts somewhere else, point the app to them:

```bash
export PNEUMO_ARTIFACTS=/path/to/artifacts      # macOS / Linux
set PNEUMO_ARTIFACTS=C:\path\to\artifacts        # Windows
```

## 2. Install and run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser. Upload a chest X-ray (JPEG/PNG) and you'll see:

- **Preprocessing view** — original → median denoised → CLAHE enhanced
- **Prediction** — Normal / Pneumonia with a confidence bar
- **Grad-CAM** — heatmap of the regions that drove the deep model's decision
- **Model performance tab** — test-set ROC-AUC for all three models
- A **model toggle** in the sidebar to switch between the deep and classical pipelines

## Notes / troubleshooting

- **First load is slow** because TensorFlow and DenseNet121 take a moment to initialize.
  After that, predictions are quick. The model is cached, so it loads only once.
- **scikit-learn / TensorFlow version warnings** can appear when loading models that were
  trained on a slightly different version (e.g. Colab). They usually still load fine; if the
  SVM fails to unpickle, install the same scikit-learn version you used in Colab
  (check with `import sklearn; sklearn.__version__` there).
- The `.keras` model uses the Keras 3 format, so you need **TensorFlow ≥ 2.16** locally.
- If you only want to demo the classical pipeline, you don't need TensorFlow installed —
  just keep `svm_pneumonia.joblib` in `artifacts/` and pick the classical model in the sidebar.

## Optional: deploy online

You can host it free on [Streamlit Community Cloud](https://streamlit.io/cloud): push
`app.py`, `requirements.txt`, and `artifacts/` to a GitHub repo and connect it. Note the
DenseNet model file is ~27 MB, which is fine for GitHub.
