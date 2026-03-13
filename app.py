import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import math
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PotatoScan AI",
    page_icon="🌿",
    layout="centered"
)

# ─────────────────────────────────────────────
# 2. CUSTOM CSS — dark botanical theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── App background ── */
.stApp {
    background: #0d1a12;
    background-image:
        radial-gradient(ellipse at 20% 50%, rgba(34,85,47,0.25) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 10%, rgba(20,60,30,0.3) 0%, transparent 55%);
}

/* ── Main container ── */
.block-container {
    max-width: 760px !important;
    padding-top: 2rem !important;
}

/* ── Hero header ── */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid rgba(120,200,130,0.15);
    margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #c8f0cc;
    letter-spacing: -0.5px;
    margin: 0;
    line-height: 1.15;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #6a9e74;
    margin-top: 0.6rem;
    font-weight: 300;
    letter-spacing: 0.3px;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1.5px dashed rgba(120,200,130,0.35) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(120,200,130,0.65) !important;
}
[data-testid="stFileUploader"] label {
    color: #8dbe95 !important;
}

/* ── Image display ── */
[data-testid="stImage"] img {
    border-radius: 14px !important;
    border: 1px solid rgba(120,200,130,0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}

/* ── Result card ── */
.result-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(120,200,130,0.2);
    border-radius: 18px;
    padding: 1.8rem 2rem;
    margin: 1.5rem 0;
    backdrop-filter: blur(6px);
}
.result-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #6a9e74;
    font-weight: 500;
    margin-bottom: 0.4rem;
}
.result-disease {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    font-weight: 600;
    margin: 0;
}
.result-healthy { color: #6ee88a; }
.result-warning { color: #f5c842; }
.result-danger  { color: #f5824a; }
.result-unknown { color: #a0a0b0; }

/* ── Confidence badge ── */
.conf-badge {
    display: inline-block;
    padding: 0.28rem 0.85rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 500;
    margin-top: 0.7rem;
    letter-spacing: 0.3px;
}
.conf-high   { background: rgba(110,232,138,0.15); color: #6ee88a; border: 1px solid rgba(110,232,138,0.35); }
.conf-medium { background: rgba(245,200,66,0.15);  color: #f5c842; border: 1px solid rgba(245,200,66,0.35);  }
.conf-low    { background: rgba(245,130,74,0.15);  color: #f5824a; border: 1px solid rgba(245,130,74,0.35);  }

/* ── Progress bars ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.55rem 0;
}
.prob-label {
    font-size: 0.82rem;
    color: #8dbe95;
    width: 200px;
    flex-shrink: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.prob-bar-bg {
    flex: 1;
    height: 8px;
    background: rgba(255,255,255,0.07);
    border-radius: 999px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}
.prob-pct {
    font-size: 0.82rem;
    color: #c8f0cc;
    width: 46px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Section heading ── */
.section-heading {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: #5a8a64;
    font-weight: 500;
    margin: 1.8rem 0 0.9rem;
    border-bottom: 1px solid rgba(120,200,130,0.1);
    padding-bottom: 0.4rem;
}

/* ── OOD warning card ── */
.ood-card {
    background: rgba(245,130,74,0.07);
    border: 1px solid rgba(245,130,74,0.3);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin: 1.5rem 0;
}
.ood-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    color: #f5824a;
    margin-bottom: 0.4rem;
}
.ood-text {
    font-size: 0.88rem;
    color: #c08070;
    line-height: 1.6;
}

/* ── Entropy meter ── */
.entropy-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-top: 0.6rem;
}
.entropy-label { font-size: 0.8rem; color: #6a9e74; width: 110px; }
.entropy-bar-bg {
    flex: 1; height: 6px;
    background: rgba(255,255,255,0.07);
    border-radius: 999px; overflow: hidden;
}
.entropy-bar-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #6ee88a, #f5c842, #f5824a);
}
.entropy-val { font-size: 0.8rem; color: #c8f0cc; width: 38px; text-align: right; }

/* ── Divider ── */
hr { border-color: rgba(120,200,130,0.1) !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 3. HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <p class="hero-title">🌿 PotatoScan AI</p>
    <p class="hero-subtitle">Upload a potato leaf image · Get instant disease diagnosis</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 4. DEVICE & MODEL
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = torch.jit.load("model_traced.pt", map_location=device)
    model.eval()
    return model

model = load_model()


# ─────────────────────────────────────────────
# 5. CONSTANTS
# ─────────────────────────────────────────────
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
DISPLAY_NAMES = {
    'Potato___Early_blight': 'Early Blight',
    'Potato___Late_blight':  'Late Blight',
    'Potato___healthy':      'Healthy Leaf',
}
DISEASE_INFO = {
    'Potato___Early_blight': "Caused by Alternaria solani. Look for dark brown spots with concentric rings (target-board pattern). Remove infected leaves and apply copper-based fungicide.",
    'Potato___Late_blight':  "Caused by Phytophthora infestans. Water-soaked lesions that turn dark brown/black. Act fast — this spreads rapidly. Apply systemic fungicides immediately.",
    'Potato___healthy':      "No signs of disease detected. Continue regular monitoring, proper irrigation, and preventative care.",
}
BAR_COLORS = {
    'Potato___Early_blight': '#f5c842',
    'Potato___Late_blight':  '#f5824a',
    'Potato___healthy':      '#6ee88a',
}

ENTROPY_THRESHOLD = 0.28
CONFIDENCE_THRESHOLD = 0.80
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# ─────────────────────────────────────────────
# 6. HELPER: entropy
# ─────────────────────────────────────────────
def compute_entropy(probs: list[float]) -> float:
    """Normalised Shannon entropy in [0, 1]."""
    n = len(probs)
    raw = -sum(p * math.log(p + 1e-9) for p in probs)
    return raw / math.log(n)


# ─────────────────────────────────────────────
# 7. FILE UPLOADER
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop a leaf image here (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)

if uploaded_file is not None:

    # ── Show image ──────────────────────────────
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    # ── Inference ───────────────────────────────
    with st.spinner("Analysing leaf…"):
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs    = model(img_tensor)
            probs_t    = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probs_t, 0)

    probs      = [probs_t[i].item() for i in range(len(CLASS_NAMES))]
    conf_val   = confidence.item()
    pred_class = CLASS_NAMES[predicted_idx.item()]
    entropy    = compute_entropy(probs)

    # ── OOD check ───────────────────────────────
    is_ood = (entropy > ENTROPY_THRESHOLD) or (conf_val < CONFIDENCE_THRESHOLD)

    if is_ood:
        st.markdown(f"""
        <div class="ood-card">
            <div class="ood-title">⚠️ Image Not Recognised</div>
            <div class="ood-text">
                This doesn't look like a potato leaf. The model is uncertain across all
                classes (uncertainty score: <strong>{entropy*100:.0f}%</strong>).<br><br>
                Please upload a clear, close-up photo of a <em>potato leaf</em> for an
                accurate diagnosis.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Determine styling ────────────────────
        if pred_class == 'Potato___healthy':
            name_class = "result-healthy"
        elif pred_class == 'Potato___Early_blight':
            name_class = "result-warning"
        else:
            name_class = "result-danger"

        if conf_val >= 0.85:
            badge_class, badge_text = "conf-high",   f"High confidence · {conf_val*100:.1f}%"
        elif conf_val >= 0.60:
            badge_class, badge_text = "conf-medium", f"Moderate confidence · {conf_val*100:.1f}%"
        else:
            badge_class, badge_text = "conf-low",    f"Low confidence · {conf_val*100:.1f}%"

        # ── Result card ──────────────────────────
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Diagnosis</div>
            <p class="result-disease {name_class}">{DISPLAY_NAMES[pred_class]}</p>
            <span class="conf-badge {badge_class}">{badge_text}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Disease info ─────────────────────────
        st.markdown(f"""
        <div class="section-heading">What this means</div>
        <p style="color:#a0c8a8; font-size:0.9rem; line-height:1.7; margin:0;">
            {DISEASE_INFO[pred_class]}
        </p>
        """, unsafe_allow_html=True)

    # ── Class probabilities (always shown) ──────
    st.markdown('<div class="section-heading">Class Probabilities</div>', unsafe_allow_html=True)

    for i, cls in enumerate(CLASS_NAMES):
        pct   = probs[i] * 100
        color = BAR_COLORS[cls]
        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-label">{DISPLAY_NAMES[cls]}</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{pct:.1f}%; background:{color};"></div>
            </div>
            <div class="prob-pct">{pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Entropy / uncertainty meter ──────────────
    st.markdown('<div class="section-heading">Model Uncertainty</div>', unsafe_allow_html=True)
    entropy_pct = entropy * 100
    threshold_pct = ENTROPY_THRESHOLD * 100
    st.markdown(f"""
    <div class="entropy-row">
        <div class="entropy-label">Uncertainty</div>
        <div class="entropy-bar-bg">
            <div class="entropy-bar-fill" style="width:{entropy_pct:.1f}%;"></div>
        </div>
        <div class="entropy-val">{entropy_pct:.0f}%</div>
    </div>
    <p style="color:#4a7a54; font-size:0.76rem; margin-top:0.4rem;">
        Threshold: {threshold_pct:.0f}% — values above this flag the image as out-of-distribution.
    </p>
    """, unsafe_allow_html=True)

else:
    # ── Empty state ──────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem; color:#3d6647;">
        <div style="font-size:3.5rem; margin-bottom:1rem;">🍃</div>
        <p style="font-size:0.95rem; line-height:1.7;">
            No image uploaded yet.<br>
            Upload a potato leaf photo above to begin your diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)