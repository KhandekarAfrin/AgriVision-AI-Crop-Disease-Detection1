from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_ID = "neghani/plantvillage-mobilenetv2"
HISTORY_DB = Path(__file__).with_name("prediction_history.db")
HISTORY_IMAGES = Path(__file__).with_name("history_images")

st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    :root { --ink:#203633; --muted:#60736f; --line:#dfe5dc; --green:#28776e; }
    .stApp, [data-testid="stAppViewContainer"] { background:#f7f5ee !important; color:var(--ink) !important; font-family:'Inter','Segoe UI',Arial,sans-serif; }
    [data-testid="stAppViewContainer"] .main, [data-testid="stAppViewContainer"] .block-container { background:#f7f5ee !important; }
    [data-testid="stMarkdownContainer"] { color:#203633 !important; }
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color:#203633; }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color:#203633 !important; }
    [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] section, [data-testid="stFileUploader"] small { color:#203633 !important; }
    [data-testid="stSidebar"] { background:#eeece3 !important; border-right:1px solid var(--line); }
    [data-testid="stHeader"] { background:rgba(247,245,238,.92) !important; }
    h1,h2,h3 { font-family:'Inter','Segoe UI',Arial,sans-serif !important; color:var(--ink); letter-spacing:-.35px; }
    .block-container { max-width:1200px; padding-top:2rem; padding-bottom:3rem; }
    .brand { display:flex; align-items:center; gap:12px; margin-bottom:20px; }
    .brand-mark { width:42px; height:42px; display:grid; place-items:center; border-radius:14px; background:#dcefe2; font-size:23px; }
    .brand-name { font:800 18px 'Inter','Segoe UI',Arial,sans-serif; color:#193b2d; letter-spacing:-.4px; }
    .brand-sub { font-size:11px; color:#718178; letter-spacing:1.5px; text-transform:uppercase; }
    .hero { position:relative; overflow:hidden; padding:34px 38px; border-radius:24px; background:linear-gradient(115deg,#173d3a 0%,#22645d 58%,#39877c 100%); color:white; margin-bottom:25px; box-shadow:0 16px 34px rgba(27,74,69,.16); }
    .hero:after { content:'✳'; position:absolute; right:48px; top:-47px; font-size:205px; color:rgba(255,255,255,.06); }
    .hero-kicker { font-size:11px; font-weight:700; letter-spacing:1.7px; text-transform:uppercase; color:#c3e7ce; }
    .hero h1 { color:white !important; margin:8px 0 7px; font-size:36px; letter-spacing:-1.1px; font-weight:700; }
    .hero p { max-width:640px; margin:0; color:#e0efe5; font-size:15px; line-height:1.65; }
    .panel { border:1px solid var(--line); background:#fffefa; border-radius:18px; padding:22px; box-shadow:0 4px 16px rgba(43,66,59,.045); }
    .panel-title { font:700 17px 'Manrope',sans-serif; margin:0 0 5px; color:var(--ink); }
    .panel-copy { font-size:13px; color:var(--muted); margin-bottom:16px; }
    .result-label { color:#6d7e74; font-size:11px; letter-spacing:1.3px; font-weight:700; text-transform:uppercase; }
    .result-name { font:700 25px 'Inter','Segoe UI',Arial,sans-serif; color:#18352b; line-height:1.28; margin:7px 0 14px; }
    .confidence-card { display:flex; align-items:center; justify-content:space-between; gap:12px; margin:16px 0 8px; padding:14px 16px; border:1px solid #d9e6e1; border-radius:13px; background:#eef5f1; color:#203633; }
    .confidence-card span { font-size:11px; font-weight:700; letter-spacing:1.1px; color:#526a65; }
    .confidence-card strong { font:700 25px 'Inter','Segoe UI',Arial,sans-serif; color:#216b62; }
    .pill { display:inline-block; border-radius:999px; padding:5px 10px; font-size:11px; font-weight:700; letter-spacing:.3px; }
    .healthy { background:#e0f3e5; color:#20633b; }
    .disease { background:#fff0e9; color:#9e492d; }
    .uncertain { background:#fff4d9; color:#85610b; }
    .advice { background:#f2f7f3; border:1px solid #e1ebe3; border-radius:13px; padding:15px 17px; margin:10px 0; }
    .advice b { color:#244d37; }
    .stButton button { border-radius:10px; }
    [data-testid="stFileUploader"] { border:1px dashed #a9c6b1; border-radius:14px; background:#f8fbf8; padding:8px; }
    @media (max-width:700px) { .hero { padding:25px 22px; } .hero h1 { font-size:29px; } .hero:after { right:-10px; } }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="brand"><div class="brand-mark">🌿</div><div><div class="brand-name">CROP DISEASE DETECTION</div><div class="brand-sub">AI-powered crop health screening</div></div></div>'
    '<div class="hero"><div class="hero-kicker">AI-powered leaf screening</div><h1>Know what your leaf is telling you.</h1>'
    '<p>Upload a clear leaf photo to check whether it appears healthy or matches a supported disease class. Get a confidence score and practical prevention guidance.</p></div>',
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


def initialize_history() -> None:
    with sqlite3.connect(HISTORY_DB) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                image_hash TEXT PRIMARY KEY,
                screened_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                alternatives TEXT NOT NULL,
                image_file TEXT NOT NULL DEFAULT ''
            )"""
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(predictions)")}
        if "image_file" not in columns:
            connection.execute("ALTER TABLE predictions ADD COLUMN image_file TEXT NOT NULL DEFAULT ''")


def save_prediction(
    image_hash: str,
    filename: str,
    prediction: str,
    confidence: float,
    alternatives: list[dict[str, str]],
    image_bytes: bytes,
) -> bool:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    HISTORY_IMAGES.mkdir(parents=True, exist_ok=True)
    image_path = HISTORY_IMAGES / f"{image_hash}{suffix}"
    image_path.write_bytes(image_bytes)
    with sqlite3.connect(HISTORY_DB) as connection:
        cursor = connection.execute(
            "INSERT OR IGNORE INTO predictions (image_hash, screened_at, filename, prediction, confidence, alternatives, image_file) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                image_hash,
                datetime.now().astimezone().isoformat(timespec="seconds"),
                filename,
                prediction,
                confidence,
                json.dumps(alternatives),
                str(image_path),
            ),
        )
        return cursor.rowcount > 0


def read_history() -> list[dict[str, str | float]]:
    with sqlite3.connect(HISTORY_DB) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT screened_at, filename, prediction, confidence, alternatives, image_file FROM predictions ORDER BY screened_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def history_csv(rows: list[dict[str, str | float]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["screened_at", "filename", "prediction", "confidence", "alternatives", "image_file"])
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def history_bundle(rows: list[dict[str, str | float]]) -> bytes:
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("screening_history.csv", history_csv(rows))
        for row in rows:
            image_file = str(row.get("image_file", ""))
            image_path = Path(image_file) if image_file else None
            if image_path and image_path.is_file():
                archive.write(image_path, f"images/{image_path.name}")
    return bundle.getvalue()


initialize_history()


def prevention_guidance(label: str) -> list[str]:
    """Provide general, low-risk prevention steps based on the predicted class name."""
    name = label.lower()
    if "healthy" in name:
        return [
            "Continue checking leaves regularly, including their undersides.",
            "Water at the soil line when practical and keep airflow around plants.",
            "Use clean tools and follow crop rotation and locally recommended varieties.",
        ]
    if any(term in name for term in ("mosaic", "yellow leaf curl", "curly top", "greening")):
        return [
            "Use resistant varieties and healthy, certified planting material where available.",
            "Remove heavily affected plants when advised by a local extension service; avoid moving sap between plants on tools or hands.",
            "Monitor the crop and nearby weeds for insect vectors, and use locally approved integrated pest management guidance.",
        ]
    if any(term in name for term in ("powdery mildew", "downy mildew", "leaf mold")):
        return [
            "Choose resistant varieties and space plants to improve airflow.",
            "Water near the soil in the morning and avoid prolonged leaf wetness where possible.",
            "Remove infected plant debris and avoid excessive nitrogen fertilization.",
        ]
    if any(term in name for term in ("rust", "blight", "spot", "scab", "rot", "anthracnose", "esca", "measles", "canker")):
        return [
            "Use healthy seed or planting stock and resistant varieties when available.",
            "Improve spacing and airflow; avoid overhead watering when practical.",
            "Remove infected leaves and crop residue, clean tools, and rotate with non-host crops where suitable.",
        ]
    if "spider mite" in name:
        return [
            "Inspect leaf undersides regularly, especially during hot, dry weather.",
            "Reduce plant stress with appropriate irrigation and avoid unnecessary broad-spectrum pesticide use.",
            "Ask a local extension service about integrated pest management options if populations increase.",
        ]
    return [
        "Use healthy planting material and resistant varieties where available.",
        "Keep good airflow, avoid prolonged leaf wetness, and clean tools between plants.",
        "Remove affected plant debris and ask a local extension service for crop-specific prevention advice.",
    ]


with st.sidebar:
    st.markdown("### About this screening")
    st.write("This app uses a pretrained MobileNetV2 image classifier for supported PlantVillage leaf classes.")
    st.caption("The model downloads from Hugging Face the first time you run the app. An internet connection is required for that initial download.")
    st.divider()
    st.markdown("**Supported scope**")
    st.caption("38 classes across 14 crops. Images outside those classes may still receive a closest-match prediction.")
    st.markdown("[Model details](https://huggingface.co/neghani/plantvillage-mobilenetv2)")
    st.divider()
    st.markdown("**Screening history**")
    st.caption("Prediction details and uploaded images are saved locally in this project folder.")
    saved_history = read_history()
    if saved_history:
        st.caption(f"{len(saved_history)} saved screening{'s' if len(saved_history) != 1 else ''}")
        st.download_button(
            "Download history + images",
            data=history_bundle(saved_history),
            file_name="crop_disease_screening_history.zip",
            mime="application/zip",
            use_container_width=True,
        )
    else:
        st.caption("Your screening history will appear here after the first prediction.")

st.markdown('<div class="panel-title">Start a leaf screening</div><div class="panel-copy">For the clearest result, photograph one leaf in natural light against a simple background.</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

if uploaded is None:
    st.markdown(
        '<div class="advice"><b>Photo tips</b><br>Include one full leaf, keep the image in focus, and avoid harsh shadows or busy backgrounds.</div>',
        unsafe_allow_html=True,
    )
else:
    try:
        image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
        image_col, result_col = st.columns([0.95, 1.05], gap="large")
        with image_col:
            st.markdown('<div class="panel-title">Uploaded image</div><div class="panel-copy">Review the image before using the screening result.</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        with st.spinner("Loading the classifier and analyzing the image… The first run may take a few minutes while the model downloads."):
            processor, model = load_model()
            inputs = processor(images=image, return_tensors="pt")
            with torch.inference_mode():
                logits = model(**inputs).logits
                probabilities = torch.softmax(logits[0].float(), dim=-1)
            if not torch.isfinite(probabilities).all() or float(probabilities.max()) <= 0:
                raise ValueError("The model returned an invalid confidence score. Please try another image.")
            scores, indices = torch.topk(probabilities, k=3)

        labels = model.config.id2label
        best_label = labels[int(indices[0])]
        best_score = float(scores[0])
        displayed_score = min(best_score, 0.999)
        readable = best_label.replace("___", " — ").replace("_", " ").strip()
        is_healthy = "healthy" in best_label.lower()
        is_uncertain = best_score < 0.50
        image_hash = hashlib.sha256(uploaded.getvalue()).hexdigest()
        alternatives = [
            {
                "class": labels[int(index)].replace("___", " — ").replace("_", " ").strip(),
                "confidence": f"{min(float(score), 0.999):.1%}",
            }
            for score, index in zip(scores.tolist(), indices.tolist())
        ]
        was_saved = save_prediction(
            image_hash,
            uploaded.name,
            readable,
            displayed_score,
            alternatives,
            uploaded.getvalue(),
        )

        with result_col:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="result-label">Screening result</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-name">{readable}</div>', unsafe_allow_html=True)
            status_text = "HEALTHY CLASS" if is_healthy else "DISEASE CLASS"
            status_style = "healthy" if is_healthy else "disease"
            if is_uncertain:
                status_text, status_style = "LOW CONFIDENCE", "uncertain"
            st.markdown(f'<span class="pill {status_style}">{status_text}</span>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="confidence-card"><span>CONFIDENCE SCORE · MAX 99.9%</span><strong>{displayed_score:.1%}</strong></div>',
                unsafe_allow_html=True,
            )
            st.progress(displayed_score)
            if is_uncertain:
                st.warning("The model is uncertain. Retake the photo in clear, even light and verify the result with a local agriculture expert.")
            elif is_healthy:
                st.success("The image most closely matches a healthy class in the model. This does not rule out conditions outside the supported classes.")
            else:
                st.warning("The image most closely matches a disease class. Treat this as a screening result, not a confirmed diagnosis.")

            st.markdown("#### Prevention guidance" if not is_healthy else "#### Keep plants healthy")
            for tip in prevention_guidance(best_label):
                st.markdown(f'<div class="advice">{tip}</div>', unsafe_allow_html=True)
            st.caption("Prediction saved to local screening history." if was_saved else "This image is already in your local screening history.")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("View alternative predictions"):
            for score, index in zip(scores.tolist(), indices.tolist()):
                label = labels[int(index)].replace("___", " — ").replace("_", " ").strip()
                st.write(f"**{label}** · {min(float(score), 0.999):.1%}")

    except Exception as exc:
        st.error("The image could not be analyzed. Check your connection and try again.")
        with st.expander("Technical details"):
            st.code(str(exc))
        st.caption("The model is downloaded from Hugging Face on first use. Confirm that the initial download has completed.")

st.divider()
with st.expander("View saved screening history"):
    saved_history = read_history()
    if saved_history:
        history_view = [
            {
                "Date": row["screened_at"],
                "Image": row["filename"],
                "Prediction": row["prediction"],
                "Confidence": f"{float(row['confidence']):.1%}",
            }
            for row in saved_history
        ]
        st.dataframe(history_view, use_container_width=True, hide_index=True)
        for row in saved_history:
            with st.expander(f"{row['prediction']} · {row['filename']} · {row['screened_at']}"):
                stored_image = Path(str(row["image_file"])) if row["image_file"] else None
                if stored_image and stored_image.exists():
                    st.image(str(stored_image), caption=str(row["filename"]), width=420)
                else:
                    st.info("No saved image is available for this history entry.")
                st.write(f"Confidence score: {float(row['confidence']):.1%}")
        st.download_button(
            "Download history bundle (CSV + images)",
            data=history_bundle(saved_history),
            file_name="crop_disease_screening_history.zip",
            mime="application/zip",
            key="history_bundle_main",
        )
    else:
        st.info("No screenings have been saved yet.")

st.caption(
    "Important: This tool supports only 38 PlantVillage classes across 14 crops. Displayed confidence scores are capped at 99.9%; they are model softmax scores, not a guarantee of diagnostic accuracy. "
    "Field conditions can differ from the training images. Confirm disease and prevention decisions with a qualified local agricultural adviser before treatment."
)
