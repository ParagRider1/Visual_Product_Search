import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import hnswlib
from ultralytics import YOLO
import cv2

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SYS.LOOKFINDER // TERMINAL",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400;1,700&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: #000000 !important;
    color: #39ff14 !important;
    font-family: 'Space Mono', monospace !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #000000 !important;
    background-image: 
        linear-gradient(rgba(0, 255, 255, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 255, 255, 0.05) 1px, transparent 1px) !important;
    background-size: 30px 30px !important;
}

/* CRT Scanline Overlay */
[data-testid="stAppViewContainer"]::after {
    content: " ";
    display: block;
    position: absolute;
    top: 0; left: 0; bottom: 0; right: 0;
    background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06));
    z-index: 9999;
    background-size: 100% 3px, 3px 100%;
    pointer-events: none;
    opacity: 0.4;
}

/* Hide Streamlit chrome */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
footer, #MainMenu,
section[data-testid="stSidebar"] { display: none !important; }

/* Main container */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Navbar HUD ── */
.navbar {
    border-bottom: 1px solid #39ff14;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2rem;
    height: 48px;
    background: rgba(0,0,0,0.9);
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 2px 15px rgba(57, 255, 20, 0.15);
}
.navbar-logo {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #0ff;
    text-shadow: 0 0 8px #0ff;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { text-shadow: 0 0 5px #0ff; }
    50% { text-shadow: 0 0 20px #0ff, 0 0 30px #0ff; }
    100% { text-shadow: 0 0 5px #0ff; }
}

.navbar-links {
    display: flex;
    gap: 2rem;
    list-style: none;
}
.navbar-links li { color: #39ff14; font-size: 0.8rem; letter-spacing: 1px; }

/* ── Hero ── */
.hero {
    padding: 4rem 2rem;
    border-bottom: 1px solid #39ff14;
    background: rgba(5, 5, 5, 0.8);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 50%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(57, 255, 20, 0.1), transparent);
    animation: scan 4s linear infinite;
}
@keyframes scan {
    0% { left: -100%; }
    100% { left: 200%; }
}

.hero-eyebrow {
    font-size: 0.8rem;
    color: #0ff;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 2px;
}
.hero-title {
    font-size: 3.5rem;
    font-weight: 700;
    color: #39ff14;
    margin-bottom: 1.5rem;
    text-shadow: 0 0 10px #39ff14;
    letter-spacing: -1px;
    line-height: 1.1;
    position: relative;
}
/* Glitch effect on hover */
.hero-title:hover {
    animation: glitch 0.2s linear infinite;
}
@keyframes glitch {
    2%, 64% { transform: translate(2px,0) skew(0deg); }
    4%, 60% { transform: translate(-2px,0) skew(0deg); }
    62% { transform: translate(0,0) skew(5deg); }
}

.hero-sub {
    font-size: 0.95rem;
    color: #0ff !important;
    -webkit-text-fill-color: #0ff !important;
    line-height: 1.6;
    text-transform: uppercase;
    max-width: 600px;
    border-left: 2px solid #39ff14;
    padding-left: 15px;
}
.cursor {
    display: inline-block;
    width: 10px;
    height: 1.2em;
    background: #0ff;
    vertical-align: middle;
    animation: blink 1s step-end infinite;
}
@keyframes blink { 50% { opacity: 0; } }

/* ── Content wrapper ── */
.content-wrap {
    max-width: 1300px;
    margin: 0 auto;
    padding: 2.5rem 2rem;
}

.hud-title {
    font-size: 1.2rem;
    color: #0ff;
    border-left: 4px solid #0ff;
    padding-left: 0.75rem;
    margin-bottom: 1.5rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    box-shadow: -10px 0 15px -10px #0ff;
    display: inline-block;
}

/* ── Upload box & HUD Corner Accents ── */
.hud-box {
    position: relative;
    padding: 4px;
    border: 1px solid rgba(0, 255, 255, 0.3);
    background: rgba(0, 20, 20, 0.4);
    backdrop-filter: blur(5px);
    transition: all 0.3s ease;
}
.hud-box:hover {
    border-color: rgba(57, 255, 20, 0.5);
    box-shadow: 0 0 20px rgba(57, 255, 20, 0.1);
}
.hud-box::before {
    content: '';
    position: absolute;
    top: -2px; left: -2px;
    width: 25px; height: 25px;
    border-top: 3px solid #0ff;
    border-left: 3px solid #0ff;
    pointer-events: none;
    z-index: 10;
    transition: border-color 0.3s;
}
.hud-box::after {
    content: '';
    position: absolute;
    bottom: -2px; right: -2px;
    width: 25px; height: 25px;
    border-bottom: 3px solid #0ff;
    border-right: 3px solid #0ff;
    pointer-events: none;
    z-index: 10;
    transition: border-color 0.3s;
}
.hud-box:hover::before, .hud-box:hover::after {
    border-color: #39ff14;
}

[data-testid="stFileUploader"] label { display: none !important; }

[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: 1px dashed #39ff14 !important;
    border-radius: 0 !important;
    padding: 4rem 2rem !important;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1rem;
}
[data-testid="stFileUploaderDropzone"] * {
    color: #39ff14 !important;
    -webkit-text-fill-color: #39ff14 !important;
    font-family: 'Space Mono', monospace !important;
    text-transform: uppercase;
}
[data-testid="stFileUploaderDropzone"] svg {
    display: none !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: #000 !important;
    color: #39ff14 !important;
    -webkit-text-fill-color: #39ff14 !important;
    border: 1px solid #39ff14 !important;
    border-radius: 0 !important;
    padding: 0.75rem 2rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px;
    box-shadow: 0 0 10px rgba(57, 255, 20, 0.4);
    margin: 10px 0 !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: #39ff14 !important;
    color: #000 !important;
    -webkit-text-fill-color: #000 !important;
    box-shadow: 0 0 20px #39ff14;
}

/* Uploaded file item bar */
[data-testid="stUploadedFile"] {
    background: rgba(0,0,0,0.8) !important;
    border: 1px solid #0ff !important;
    border-radius: 0 !important;
}
[data-testid="stUploadedFile"] * {
    color: #0ff !important;
    -webkit-text-fill-color: #0ff !important;
    font-family: 'Space Mono', monospace !important;
}
[data-testid="stUploadedFile"] button {
    background: #000 !important;
}

/* ── Thin card (image stages) ── */
.stage-card {
    background: rgba(0,0,0,0.6);
    border: 1px solid #39ff14;
    border-radius: 0;
    position: relative;
}
.stage-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    box-shadow: inset 0 0 20px rgba(57,255,20,0.1);
    pointer-events: none;
}
.stage-card-label {
    font-size: 0.7rem;
    color: #000;
    background: #39ff14;
    padding: 0.3rem 0.6rem;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 1px;
}
.stage-card-caption {
    font-size: 0.75rem;
    color: #0ff;
    padding: 0.75rem;
    border-top: 1px dotted #39ff14;
    display: flex;
    justify-content: space-between;
}

/* ── Search button ── */
div[data-testid="stButton"] > button {
    background: rgba(0,0,0,0.8) !important;
    color: #39ff14 !important;
    border: 1px solid #39ff14 !important;
    border-radius: 0 !important;
    padding: 1.25rem 2rem !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    width: 100% !important;
    box-shadow: 0 0 15px rgba(57, 255, 20, 0.2) !important;
    transition: all 0.2s;
    letter-spacing: 2px;
}
div[data-testid="stButton"] > button:hover {
    background: #39ff14 !important;
    color: #000 !important;
    box-shadow: 0 0 30px rgba(57, 255, 20, 0.8) !important;
}
div[data-testid="stButton"] > button::before {
    content: "> EXECUTE_";
}

/* ── Results grid ── */
.product-card {
    background: rgba(0,0,0,0.8);
    border: 1px solid #0ff;
    border-radius: 0;
    padding: 3px;
    position: relative;
    transition: all 0.3s;
}
.product-card:hover {
    border-color: #39ff14;
    box-shadow: 0 0 15px rgba(57, 255, 20, 0.4);
    transform: translateY(-2px);
    z-index: 10;
}
.product-rank {
    font-size: 0.7rem;
    color: #000;
    background: #0ff;
    padding: 0.3rem 0.5rem;
    text-transform: uppercase;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 3px;
    letter-spacing: 1px;
}
.product-card:hover .product-rank {
    background: #39ff14;
}
.product-score-tag {
    color: #39ff14;
    font-size: 0.85rem;
    font-weight: 700;
    padding: 0.5rem 0.2rem 0.2rem;
    display: block;
    text-align: right;
    letter-spacing: 1px;
}
.product-bar-wrap {
    height: 6px;
    background: #000;
    border: 1px solid #39ff14;
    margin: 0 2px 2px;
}
.product-bar {
    height: 100%;
    background: repeating-linear-gradient(
      45deg,
      #39ff14,
      #39ff14 4px,
      #000 4px,
      #000 8px
    );
    box-shadow: 0 0 8px #39ff14;
}

/* ── Stats row ── */
.stat-card {
    border: 1px dashed #39ff14;
    padding: 1.5rem;
    background: rgba(57, 255, 20, 0.05);
    backdrop-filter: blur(2px);
    position: relative;
}
.stat-card::after {
    content: '';
    position: absolute;
    bottom: 0; right: 0;
    width: 10px; height: 10px;
    background: #39ff14;
}
.stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: #39ff14;
    text-shadow: 0 0 10px #39ff14;
    margin-bottom: 0.5rem;
}
.stat-label {
    font-size: 0.75rem;
    color: #0ff;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Images ── */
[data-testid="stImage"] img { 
    border-radius: 0 !important; 
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: rgba(0,0,0,0.9) !important;
    border: 1px solid #ff003c !important;
    border-left: 5px solid #ff003c !important;
    border-radius: 0 !important;
    color: #ff003c !important;
    box-shadow: 0 0 10px rgba(255,0,60,0.2);
}

/* ── Spinner ── */
[data-testid="stSpinner"] p { color: #39ff14 !important; font-family: 'Space Mono', monospace !important; font-weight: bold !important; letter-spacing: 1px !important;}
[data-testid="stSpinner"] i { border-color: #39ff14 !important; border-bottom-color: transparent !important; }

</style>
""", unsafe_allow_html=True)


# ── Navbar ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="navbar">
    <div class="navbar-logo">SYS.LOOKFINDER // V2.0</div>
    <ul class="navbar-links">
        <li>[ PORT_1 : SEARCH ]</li>
        <li>[ PORT_2 : DB_GALLERY ]</li>
        <li><span class="cursor" style="height:1em;width:6px;background:#39ff14;"></span></li>
    </ul>
</div>
""", unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="content-wrap" style="padding: 0;">
        <p class="hero-eyebrow">>> INITIALIZING NEURAL UPLINK...</p>
        <h1 class="hero-title">IMAGE-BASED PRODUCT<br>DISCOVERY SYSTEM</h1>
        <p class="hero-sub">
            > AWAITING IMAGE INPUT...<br>
            > TARGET MATCH IDENTIFICATION ACTIVE.<br>
            > CONNECTED TO DEEPFASHION MAINFRAME.<span class="cursor"></span>
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR       = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
INDEX_DIR      = BASE_DIR / "indexes"
DATASET_ROOT   = BASE_DIR / "data" / "DeepFashion" / "Img"


# ── Load models ───────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_clip():
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return m, p

@st.cache_resource(show_spinner=False)
def load_yolo():
    return YOLO("yolov8n.pt")

with st.spinner(">> BOOTING NEURAL NETWORKS..."):
    model, processor = load_clip()
    yolo_model       = load_yolo()


# ── Load embeddings & index ───────────────────────────────────────────────────

embeddings_path = EMBEDDINGS_DIR / "gallery_embeddings_full.npy"
metadata_path   = EMBEDDINGS_DIR / "gallery_metadata_full.csv"
index_path      = INDEX_DIR      / "fashion_hnsw_full.index"

for path, label in [
    (embeddings_path, "EMBED_TENSOR"),
    (metadata_path,   "META_DB"),
    (index_path,      "HNSW_INDEX"),
]:
    if not path.exists():
        st.error(f">> ERROR: MISSING MODULE {label} AT {path}")
        st.stop()

with st.spinner(">> MOUNTING DB FRAGMENTS..."):
    embeddings = np.load(embeddings_path)
    metadata   = pd.read_csv(metadata_path)
    dim        = embeddings.shape[1]
    index      = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(str(index_path))
    index.set_ef(50)

n_gallery = len(metadata)


# ── Upload Section ────────────────────────────────────────────────────────────

st.markdown("""
<div class="content-wrap" style="padding-bottom: 0;">
    <h2 class="hud-title">INPUT_INTERFACE // UPLOAD</h2>
</div>
""", unsafe_allow_html=True)

_, upload_col, _ = st.columns([1, 3, 1])
with upload_col:
    st.markdown('<div class="hud-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

if uploaded_file is not None:

    image    = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # ── YOLO Detection ──
    with st.spinner(">> EXECUTING YOLO_V8 SCAN..."):
        results = yolo_model(image_np)
    boxes = results[0].boxes

    if len(boxes) == 0:
        st.error(">> ERR_NO_TARGET_DETECTED. ABORTING.")
        st.stop()

    box = boxes[0]
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    conf = float(box.conf[0].cpu().numpy()) if hasattr(box, "conf") else None

    # Detection overlay (Cyberpunk style)
    det_img = image_np.copy()
    cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    # Target crosshairs
    cv2.line(det_img, (x1, y1), (x1+20, y1), (57, 255, 20), 3)
    cv2.line(det_img, (x1, y1), (x1, y1+20), (57, 255, 20), 3)
    cv2.line(det_img, (x2, y2), (x2-20, y2), (57, 255, 20), 3)
    cv2.line(det_img, (x2, y2), (x2, y2-20), (57, 255, 20), 3)

    crop     = image_np[y1:y2, x1:x2]
    crop_pil = Image.fromarray(crop)

    # ── 3-stage image row ──
    st.markdown("""
    <div class="content-wrap" style="padding-bottom: 1rem;">
        <h2 class="hud-title">TARGET_ACQUISITION // STATUS_OK</h2>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="stage-card"><div class="stage-card-label">RAW_INPUT</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('<div class="stage-card-caption"><span>> SRC_IMG</span><span>[OK]</span></div></div>', unsafe_allow_html=True)

    with c2:
        conf_txt = f"{conf:.2%}" if conf else "LOCATED"
        st.markdown(f'<div class="stage-card"><div class="stage-card-label">BOUNDING_BOX</div>', unsafe_allow_html=True)
        st.image(det_img, channels="RGB", use_container_width=True)
        st.markdown(f'<div class="stage-card-caption"><span>> CONFIDENCE</span><span>[{conf_txt}]</span></div></div>', unsafe_allow_html=True)

    with c3:
        h, w = crop.shape[:2]
        st.markdown('<div class="stage-card"><div class="stage-card-label">ISOLATED_CROP</div>', unsafe_allow_html=True)
        st.image(crop_pil, use_container_width=True)
        st.markdown(f'<div class="stage-card-caption"><span>> DIMENSIONS</span><span>[{w}x{h}]</span></div></div>', unsafe_allow_html=True)

    # ── Search Button ──
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        confirm = st.button("SEARCH", key="search_btn")

    # ── Retrieval ────────────────────────────────────────────────────────────

    if confirm:

        with st.spinner(">> QUERYING HNSW_INDEX..."):
            inputs = processor(images=crop_pil, return_tensors="pt")
            with torch.no_grad():
                query_emb = model.get_image_features(**inputs)
            if not isinstance(query_emb, torch.Tensor):
                query_emb = query_emb.pooler_output
            query_emb = F.normalize(query_emb, p=2, dim=-1).cpu().numpy()
            labels, distances = index.knn_query(query_emb, k=5)

        # ── Results header ──
        st.markdown("""
        <div class="content-wrap" style="padding-bottom: 1rem; margin-top:2rem;">
            <h2 class="hud-title">QUERY_RESULTS // MATCHES_FOUND</h2>
        </div>
        """, unsafe_allow_html=True)

        result_cols = st.columns(5)
        valid_sims = []

        for i in range(5):
            idx        = labels[0][i]
            similarity = float(1 - distances[0][i])
            valid_sims.append(similarity)
            rel_path   = metadata.iloc[idx]["image_path"]
            img_path   = DATASET_ROOT / rel_path
            bar_pct    = int(similarity * 100)

            with result_cols[i]:
                if not img_path.exists():
                    st.markdown(f"""
                    <div class="product-card" style="padding:2rem;text-align:center;">
                        <div style="font-size:2rem;color:#ff003c;">[X]</div>
                        <div style="font-size:0.75rem;margin-top:.5rem;color:#0ff;">ERR_404</div>
                    </div>""", unsafe_allow_html=True)
                    continue

                result_img = Image.open(img_path).convert("RGB")

                st.markdown(f"""
                <div class="product-card">
                    <div class="product-rank">MATCH_{i+1}</div>
                """, unsafe_allow_html=True)

                st.image(result_img, use_container_width=True)

                st.markdown(f"""
                    <span class="product-score-tag">SIM: {similarity:.2%}</span>
                    <div class="product-bar-wrap">
                        <div class="product-bar" style="width:{bar_pct}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Stats ──
        avg_sim  = sum(valid_sims) / len(valid_sims)
        best_sim = max(valid_sims)

        st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        for col, label, value in [
            (s1, "MAX_FIDELITY", f"{best_sim:.2%}"),
            (s2, "AVG_VARIANCE", f"{avg_sim:.2%}"),
            (s3, "DB_ENTRIES",   f"{n_gallery:,}"),
        ]:
            with col:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{value}</div>
                    <div class="stat-label">>> {label}</div>
                </div>
                """, unsafe_allow_html=True)