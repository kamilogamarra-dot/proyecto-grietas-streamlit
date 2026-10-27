from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError


st.set_page_config(
    page_title="Inspección de grietas",
    page_icon="▦",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');
    :root {
        --ink: #202523;
        --muted: #68716d;
        --paper: #f4f5f1;
        --line: #d9ded8;
        --accent: #e87836;
        --green: #557a55;
    }
    html, body, [class*="st-"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1180px; padding-top: 2.2rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    .eyebrow { color: var(--accent); font: 500 0.75rem 'DM Mono', monospace; text-transform: uppercase; }
    .app-title { font-size: 2.35rem; font-weight: 700; margin: 0.25rem 0 0.4rem; }
    .app-subtitle { color: var(--muted); margin: 0 0 1.7rem; }
    .metric-panel {
        border-top: 2px solid var(--ink); padding: 1rem 0 0.7rem;
        min-height: 106px;
    }
    .metric-label { color: var(--muted); font-size: 0.82rem; }
    .metric-value { color: var(--ink); font-size: 1.65rem; font-weight: 700; margin-top: 0.35rem; }
    .risk-high { color: #b34425; }
    .risk-medium { color: #a45d17; }
    .risk-low { color: var(--green); }
    .note { border-left: 3px solid var(--accent); padding: 0.65rem 0.9rem; color: var(--muted); background: #ebece7; }
    div[data-testid="stFileUploader"] { border: 1px dashed #aab2ab; border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Cargando modelo...")
def load_model(model_path: str):
    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


def detect_edges(image: Image.Image) -> tuple[np.ndarray, float, int]:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        100,
        minLineLength=50,
        maxLineGap=10,
    )

    angles: list[float] = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = np.asarray(line).ravel()[:4]
            dx = abs(int(x2) - int(x1))
            dy = abs(int(y2) - int(y1))
            angles.append(float(np.degrees(np.arctan2(dx, dy))))

    inclination = float(np.mean(angles)) if angles else 0.0
    return edges, inclination, len(angles)


def classify_image(model, image: Image.Image) -> float:
    resized = image.convert("RGB").resize((227, 227))
    batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    prediction = np.asarray(model.predict(batch, verbose=0)).reshape(-1)
    if prediction.size == 0:
        raise ValueError("El modelo no devolvió una predicción.")
    return float(prediction[0])


def risk_level(has_crack: bool, inclination: float) -> tuple[str, str]:
    if has_crack and inclination > 3.0:
        return "ALTO", "risk-high"
    if has_crack or inclination > 1.5:
        return "MEDIO", "risk-medium"
    return "BAJO", "risk-low"


st.markdown('<div class="eyebrow">Análisis visual · Concreto</div>', unsafe_allow_html=True)
st.markdown('<h1 class="app-title">Inspección de grietas</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Carga una fotografía para analizar bordes, inclinación y clasificación.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Modelo de clasificación")
    model_upload = st.file_uploader("Archivo Keras (.keras)", type=["keras"])
    st.caption("También puedes dejar modelo_base_grietas.keras junto a app.py.")

    local_model = Path(__file__).resolve().parent / "modelo_base_grietas.keras"
    model_path: Path | None = local_model if local_model.is_file() else None
    if model_upload is not None:
        model_bytes = model_upload.getvalue()
        model_hash = hashlib.sha256(model_bytes).hexdigest()
        model_path = Path(tempfile.gettempdir()) / f"grietas_{model_hash}.keras"
        if not model_path.exists():
            model_path.write_bytes(model_bytes)

    if model_path is None:
        st.info("No hay modelo cargado. El análisis de bordes seguirá disponible.")
    else:
        st.success(f"Modelo listo: {model_path.name}")

uploaded_image = st.file_uploader(
    "Fotografía de la superficie",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    key="image_upload",
)

if uploaded_image is None:
    st.markdown(
        '<div class="note">Selecciona una imagen JPG, PNG, BMP o WEBP para comenzar el análisis.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

try:
    image = Image.open(uploaded_image).convert("RGB")
except (UnidentifiedImageError, OSError):
    st.error("No se pudo abrir la imagen. Prueba con otro archivo.")
    st.stop()

edges, inclination, line_count = detect_edges(image)
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.subheader("Imagen")
    st.image(image, use_container_width=True)
    st.caption(f"{image.width} × {image.height} px")

with right:
    st.subheader("Bordes detectados")
    st.image(edges, clamp=True, use_container_width=True)

    metric_one, metric_two = st.columns(2)
    with metric_one:
        st.markdown(
            f'<div class="metric-panel"><div class="metric-label">Inclinación estimada</div>'
            f'<div class="metric-value">{inclination:.2f}°</div></div>',
            unsafe_allow_html=True,
        )
    with metric_two:
        st.markdown(
            f'<div class="metric-panel"><div class="metric-label">Segmentos lineales</div>'
            f'<div class="metric-value">{line_count}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    if model_path is None:
        st.subheader("Clasificación CNN")
        st.write("Pendiente: carga el archivo `.keras` generado por el notebook.")
    else:
        try:
            model = load_model(str(model_path))
            probability = classify_image(model, image)
            has_crack = probability > 0.5
            label = "Se detectó grieta" if has_crack else "No se detectó grieta"
            risk, risk_class = risk_level(has_crack, inclination)

            st.subheader("Resultado")
            st.write(label)
            st.progress(min(max(probability, 0.0), 1.0), text=f"Probabilidad de grieta: {probability:.1%}")
            st.markdown(
                f'<div class="metric-panel"><div class="metric-label">Nivel de riesgo estimado</div>'
                f'<div class="metric-value {risk_class}">{risk}</div></div>',
                unsafe_allow_html=True,
            )
        except ModuleNotFoundError as error:
            if error.name == "tensorflow":
                st.warning(
                    "La clasificación CNN necesita TensorFlow. Este entorno usa Python 3.14, "
                    "que no es compatible con la instalación disponible. El análisis de bordes "
                    "sigue activo."
                )
                with st.expander("Habilitar clasificación CNN"):
                    st.markdown(
                        "Instala Python 3.11 y, desde esta carpeta, ejecuta:\n\n"
                        "```powershell\n"
                        "py -3.11 -m venv .venv\n"
                        ".venv\\Scripts\\Activate.ps1\n"
                        "python -m pip install -r requirements-model.txt\n"
                        "streamlit run app.py\n"
                        "```"
                    )
            else:
                st.error(f"Falta un módulo necesario: {error.name}")
        except Exception as error:
            st.error(f"No se pudo cargar o ejecutar el modelo: {error}")

st.caption(
    "La inclinación se estima a partir de segmentos detectados con Canny y Hough; "
    "es una señal visual preliminar, no una evaluación estructural."
)