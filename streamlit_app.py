import os
import json
import tempfile
import numpy as np
import cv2
import pandas as pd
import streamlit as st

from config import METRICS_FILE, SPACES_FILE, REPORTS_DIR
from dataset import read_yolo_labels
from predictor import ParkingPredictor, annotate
from spaces import load_spaces

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Park Sense AI — Parking Space Occupancy Detection",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Custom Theme & CSS Design System (Linear / Vercel Modern Dashboard Style)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Base CSS Variables for light & dark resilience */
:root {
    --ps-primary: #0ea5e9;
    --ps-primary-dark: #0284c7;
    --ps-primary-glow: rgba(14, 165, 233, 0.25);
    --ps-success: #10b981;
    --ps-success-bg: rgba(16, 185, 129, 0.12);
    --ps-danger: #ef4444;
    --ps-danger-bg: rgba(239, 68, 68, 0.12);
    --ps-warning: #f59e0b;
    --ps-warning-bg: rgba(245, 158, 11, 0.12);
    --ps-card-bg: rgba(128, 128, 128, 0.04);
    --ps-card-border: rgba(128, 128, 128, 0.14);
    --ps-card-hover: rgba(128, 128, 128, 0.07);
    --ps-radius-lg: 16px;
    --ps-radius-md: 12px;
    --ps-radius-sm: 8px;
}

/* Remove default excess top padding */
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1300px;
}

/* Hero & Header Section */
.ps-hero-container {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.08) 0%, rgba(99, 102, 241, 0.04) 100%);
    border: 1px solid var(--ps-card-border);
    border-radius: var(--ps-radius-lg);
    padding: 24px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.04);
}

.ps-hero-brand {
    display: flex;
    align-items: center;
    gap: 16px;
}

.ps-logo-badge {
    width: 50px;
    height: 50px;
    background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 100%);
    color: #ffffff;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    font-weight: 800;
    box-shadow: 0 8px 16px var(--ps-primary-glow);
}

.ps-hero-title {
    font-size: 1.65rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.ps-hero-subtitle {
    font-size: 0.92rem;
    opacity: 0.78;
    margin-top: 4px;
    margin-bottom: 0;
    line-height: 1.4;
}

.ps-hero-badges {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}

.ps-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    background: var(--ps-card-bg);
    border: 1px solid var(--ps-card-border);
}

.ps-pill-primary {
    background: rgba(14, 165, 233, 0.1);
    color: var(--ps-primary);
    border-color: rgba(14, 165, 233, 0.25);
}

.ps-pill-success {
    background: var(--ps-success-bg);
    color: var(--ps-success);
    border-color: rgba(16, 185, 129, 0.25);
}

.ps-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: currentColor;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background-color: var(--ps-card-bg);
    padding: 6px;
    border-radius: var(--ps-radius-md);
    border: 1px solid var(--ps-card-border);
    margin-bottom: 24px;
}

.stTabs [data-baseweb="tab"] {
    height: 42px;
    border-radius: var(--ps-radius-sm);
    padding: 8px 24px;
    font-weight: 600;
    font-size: 0.92rem;
    border: none !important;
    transition: all 0.2s ease-in-out;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--ps-primary-dark) 0%, var(--ps-primary) 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px var(--ps-primary-glow);
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}

/* Card Containers */
.ps-card {
    background: var(--ps-card-bg);
    border: 1px solid var(--ps-card-border);
    border-radius: var(--ps-radius-lg);
    padding: 22px;
    margin-bottom: 20px;
    transition: border-color 0.2s;
    box-shadow: 0 2px 12px -2px rgba(0, 0, 0, 0.03);
}

.ps-card:hover {
    border-color: rgba(14, 165, 233, 0.35);
}

.ps-section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--ps-card-border);
}

.ps-section-title {
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0;
}

.ps-step-badge {
    background: var(--ps-primary);
    color: #ffffff;
    font-size: 0.72rem;
    font-weight: 800;
    padding: 2px 8px;
    border-radius: 6px;
    letter-spacing: 0.04em;
}

/* Metrics Grid & Cards */
.ps-metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin: 16px 0 20px 0;
}

.ps-metric-tile {
    background: var(--ps-card-bg);
    border: 1px solid var(--ps-card-border);
    border-radius: var(--ps-radius-md);
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}

.ps-metric-tile:hover {
    transform: translateY(-2px);
    border-color: rgba(14, 165, 233, 0.35);
}

.ps-metric-accent-bar {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
}

.ps-accent-total { background: linear-gradient(90deg, #38bdf8, #818cf8); }
.ps-accent-free { background: linear-gradient(90deg, #10b981, #34d399); }
.ps-accent-occ { background: linear-gradient(90deg, #ef4444, #f87171); }
.ps-accent-rate { background: linear-gradient(90deg, #f59e0b, #fbbf24); }

.ps-metric-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
}

.ps-metric-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    opacity: 0.72;
}

.ps-metric-icon-circle {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
}

.ps-metric-value {
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.02em;
}

.ps-metric-subtext {
    font-size: 0.8rem;
    opacity: 0.7;
    margin-top: 6px;
}

.ps-progress-track {
    width: 100%;
    height: 6px;
    background: rgba(128, 128, 128, 0.15);
    border-radius: 999px;
    margin-top: 10px;
    overflow: hidden;
}

.ps-progress-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.4s ease;
}

/* Agreement Banner Pill */
.ps-agreement-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    padding: 12px 18px;
    border-radius: var(--ps-radius-md);
    margin-bottom: 18px;
    font-size: 0.9rem;
}

.ps-agreement-high {
    background: var(--ps-success-bg);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: var(--ps-success);
}

.ps-agreement-alert {
    background: var(--ps-warning-bg);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: var(--ps-warning);
}

/* Legend Bar */
.ps-legend-container {
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
    background: var(--ps-card-bg);
    border: 1px solid var(--ps-card-border);
    border-radius: var(--ps-radius-sm);
    padding: 10px 16px;
    margin-bottom: 16px;
    font-size: 0.82rem;
}

.ps-legend-title {
    font-weight: 700;
    letter-spacing: 0.04em;
    opacity: 0.7;
    text-transform: uppercase;
    font-size: 0.72rem;
}

.ps-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-weight: 600;
}

.ps-legend-indicator {
    width: 12px;
    height: 12px;
    border-radius: 3px;
    display: inline-block;
}

.ps-ind-free { background-color: #10b981; }
.ps-ind-occ { background-color: #ef4444; }
.ps-ind-diff { background-color: #f59e0b; border: 1px dashed #ffffff; }

/* Warning Chip */
.ps-warning-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    background: var(--ps-warning-bg);
    border: 1px solid rgba(245, 158, 11, 0.25);
    color: var(--ps-warning);
    border-radius: var(--ps-radius-sm);
    font-size: 0.82rem;
    font-weight: 600;
    margin-top: 10px;
    margin-bottom: 16px;
}

/* Model Report Styled Stat Card */
.ps-report-card {
    background: var(--ps-card-bg);
    border: 1px solid var(--ps-card-border);
    border-radius: var(--ps-radius-md);
    padding: 20px;
    height: 100%;
}

.ps-report-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin-top: 14px;
}

.ps-mini-stat {
    background: rgba(128, 128, 128, 0.05);
    border: 1px solid var(--ps-card-border);
    border-radius: var(--ps-radius-sm);
    padding: 12px 14px;
}

.ps-mini-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    opacity: 0.7;
}

.ps-mini-val {
    font-size: 1.35rem;
    font-weight: 800;
    margin-top: 2px;
    font-family: 'JetBrains Mono', monospace;
}

/* Custom Table Styling */
.ps-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    margin-top: 12px;
}

.ps-table th {
    text-align: left;
    padding: 10px 14px;
    background: rgba(128, 128, 128, 0.08);
    font-weight: 700;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 1px solid var(--ps-card-border);
}

.ps-table td {
    padding: 12px 14px;
    border-bottom: 1px solid var(--ps-card-border);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}

.ps-table tr:hover td {
    background: var(--ps-card-hover);
}

/* Primary Button & Download Button Styling */
div.stDownloadButton > button, div.stButton > button {
    border-radius: var(--ps-radius-sm) !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s ease !important;
    padding: 8px 20px !important;
}

div.stDownloadButton > button:hover, div.stButton > button:hover {
    box-shadow: 0 4px 14px var(--ps-primary-glow) !important;
    transform: translateY(-1px) !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cached Model Initializer
# -----------------------------------------------------------------------------
@st.cache_resource
def get_predictor():
    return ParkingPredictor()


# -----------------------------------------------------------------------------
# Product Header / Hero Section
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="ps-hero-container">
        <div class="ps-hero-brand">
            <div class="ps-logo-badge">🅿️</div>
            <div>
                <h1 class="ps-hero-title">Park Sense AI</h1>
                <p class="ps-hero-subtitle">Intelligent Parking Space Occupancy Detection & Real-Time Lot Analytics</p>
            </div>
        </div>
        <div class="ps-hero-badges">
            <span class="ps-pill ps-pill-success"><span class="ps-dot"></span> Engine Online</span>
            <span class="ps-pill">HOG + HSV + Texture</span>
            <span class="ps-pill ps-pill-primary">RBF SVM (PKLot Trained)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_detect, tab_model = st.tabs(["🔍 Detect & Analyze", "📊 Model Performance Report"])


# =============================================================================
# TAB 1: DETECT & ANALYZE
# =============================================================================
with tab_detect:
    # ---------------- Step 1: Input & Configuration Card ----------------
    st.markdown(
        """
        <div class="ps-card">
            <div class="ps-section-header">
                <div class="ps-section-title">
                    <span class="ps-step-badge">STEP 1</span>
                    <span>Input Configuration & Spatial Metadata</span>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        img_file = st.file_uploader(
            "📷 Upload Parking Lot Camera Image",
            type=["jpg", "jpeg", "png"],
            help="High-resolution overhead or angle camera snapshot",
        )
    with c2:
        lbl_file = st.file_uploader(
            "📄 Space Boxes (YOLO .txt) — Optional",
            type=["txt"],
            help="If omitted, predefined camera geometry from spaces.json will be utilized",
        )

    compare = st.checkbox(
        "⚡ Highlight Disagreements vs Ground Truth Label File",
        value=False,
        help="Outlines spaces in yellow where the AI prediction contradicts the uploaded label file",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Step 2: Inference & Results Processing ----------------
    if img_file:
        img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
        truth = None

        if lbl_file:
            with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as f:
                f.write(lbl_file.read())
            spaces, truth = read_yolo_labels(f.name, img.shape[1], img.shape[0])
            os.remove(f.name)
        else:
            spaces = load_spaces(SPACES_FILE)

        if not spaces:
            st.warning("⚠️ No parking spaces defined. Please upload a YOLO label file or configure spaces.json using define_spaces.py.")
        else:
            # Predict & Annotate
            labels, conf = get_predictor().predict(img, spaces)
            out, s = annotate(img, spaces, labels, truth if compare else None)

            st.markdown(
                """
                <div class="ps-card">
                    <div class="ps-section-header">
                        <div class="ps-section-title">
                            <span class="ps-step-badge" style="background:#10b981;">STEP 2</span>
                            <span>Live Occupancy Telemetry & Analytics</span>
                        </div>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            # Metric Cards Display
            total_cnt = s["total"]
            free_cnt = s["free"]
            occ_cnt = s["occupied"]
            occ_pct = s["occupancy_pct"]
            free_pct = round((free_cnt / total_cnt * 100), 1) if total_cnt > 0 else 0.0

            # Progress Bar Color Calculation
            fill_color = "#ef4444" if occ_pct > 80 else ("#f59e0b" if occ_pct > 50 else "#10b981")

            st.markdown(
                f"""
                <div class="ps-metrics-grid">
                    <div class="ps-metric-tile">
                        <div class="ps-metric-accent-bar ps-accent-total"></div>
                        <div class="ps-metric-top">
                            <span class="ps-metric-label">Total Spaces</span>
                            <div class="ps-metric-icon-circle" style="background:rgba(56, 189, 248, 0.12); color:#0284c7;">🅿️</div>
                        </div>
                        <div class="ps-metric-value">{total_cnt}</div>
                        <div class="ps-metric-subtext">Configured lot capacity</div>
                    </div>
                    <div class="ps-metric-tile">
                        <div class="ps-metric-accent-bar ps-accent-free"></div>
                        <div class="ps-metric-top">
                            <span class="ps-metric-label">Available Free</span>
                            <div class="ps-metric-icon-circle" style="background:rgba(16, 185, 129, 0.12); color:#10b981;">🟢</div>
                        </div>
                        <div class="ps-metric-value" style="color:#10b981;">{free_cnt}</div>
                        <div class="ps-metric-subtext">{free_pct}% open capacity</div>
                    </div>
                    <div class="ps-metric-tile">
                        <div class="ps-metric-accent-bar ps-accent-occ"></div>
                        <div class="ps-metric-top">
                            <span class="ps-metric-label">Occupied</span>
                            <div class="ps-metric-icon-circle" style="background:rgba(239, 68, 68, 0.12); color:#ef4444;">🚗</div>
                        </div>
                        <div class="ps-metric-value" style="color:#ef4444;">{occ_cnt}</div>
                        <div class="ps-metric-subtext">{occ_pct:.1f}% occupied capacity</div>
                    </div>
                    <div class="ps-metric-tile">
                        <div class="ps-metric-accent-bar ps-accent-rate"></div>
                        <div class="ps-metric-top">
                            <span class="ps-metric-label">Occupancy Rate</span>
                            <div class="ps-metric-icon-circle" style="background:rgba(245, 158, 11, 0.12); color:#f59e0b;">📊</div>
                        </div>
                        <div class="ps-metric-value">{occ_pct:.0f}%</div>
                        <div class="ps-progress-track">
                            <div class="ps-progress-fill" style="width:{occ_pct}%; background:{fill_color};"></div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Ground Truth Agreement Badge
            if truth is not None and len(truth) == len(labels):
                acc = float((np.array(truth) == labels).mean() * 100)
                diff_cnt = int((np.array(truth) != labels).sum())
                if acc >= 98.0:
                    badge_cls = "ps-agreement-high"
                    badge_icon = "✅"
                    badge_text = f"Exceptional Agreement with Ground Truth: <strong>{acc:.2f}%</strong> ({diff_cnt} spaces differ)"
                else:
                    badge_cls = "ps-agreement-alert"
                    badge_icon = "⚠️"
                    badge_text = f"Label Discrepancies Identified: Agreement is <strong>{acc:.2f}%</strong> ({diff_cnt} spaces differ)"

                st.markdown(
                    f"""
                    <div class="ps-agreement-banner {badge_cls}">
                        <div>{badge_icon} {badge_text}</div>
                        <span style="font-size:0.8rem; opacity:0.85;">Labels audited via YOLO format</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Interactive Legend Bar
            diff_pill_html = '<span class="ps-legend-item"><span class="ps-legend-indicator ps-ind-diff"></span> Disagreement with Label</span>' if compare else ""
            st.markdown(
                f"""
                <div class="ps-legend-container">
                    <span class="ps-legend-title">Overlay Legend:</span>
                    <span class="ps-legend-item"><span class="ps-legend-indicator ps-ind-free"></span> Free Space (Available)</span>
                    <span class="ps-legend-item"><span class="ps-legend-indicator ps-ind-occ"></span> Occupied Space</span>
                    {diff_pill_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Annotated Image Canvas
            st.image(
                cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                use_container_width=True,
                caption=f"Visual Detection Segmentation · Image Dimensions: {img.shape[1]}×{img.shape[0]} px",
            )

            # Low Confidence Warning Chips
            low = [i + 1 for i, c in enumerate(conf) if c < 0.75]
            if low:
                st.markdown(
                    f"""
                    <div class="ps-warning-chip">
                        <span>⚠️ <strong>Low-Confidence Inferences (<75%):</strong> Space IDs {low}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Export Button
            ok, buf = cv2.imencode(".jpg", out)
            c_down1, _ = st.columns([1, 2])
            with c_down1:
                st.download_button(
                    label="📥 Download Annotated Image (JPEG)",
                    data=buf.tobytes(),
                    file_name="parksense_annotated_result.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div class="ps-card" style="text-align: center; padding: 48px 24px;">
                <div style="font-size: 42px; margin-bottom: 12px; opacity: 0.8;">🅿️</div>
                <h3 style="font-size: 1.2rem; font-weight: 700; margin: 0 0 6px 0;">No Image Selected</h3>
                <p style="font-size: 0.9rem; opacity: 0.7; max-width: 500px; margin: 0 auto;">
                    Upload a parking lot overhead photo in Step 1 to run real-time space segmentation, feature extraction, and occupancy classification.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# TAB 2: MODEL PERFORMANCE REPORT
# =============================================================================
with tab_model:
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as f:
            m = json.load(f)

        selected_model_name = m.get("selected_model", "rbf_svm_C10")
        train_spaces_cnt = m.get("train_spaces", 54048)
        feature_dim = m.get("feature_dim", 1792)
        latency = m.get("inference_ms_per_space", 0.22)

        # Overview Card
        st.markdown(
            f"""
            <div class="ps-card">
                <div class="ps-section-header">
                    <div class="ps-section-title">
                        <span>🏆 Selected Classifier & Architecture</span>
                    </div>
                    <span class="ps-pill ps-pill-primary">{selected_model_name} (Active Pipeline)</span>
                </div>
                <div style="display: flex; gap: 20px; flex-wrap: wrap; font-size: 0.88rem; opacity: 0.85;">
                    <div><strong>Architecture:</strong> StandardScaler ➔ PCA(150) ➔ RBF SVM (C=10)</div>
                    <div><strong>Training Samples:</strong> {train_spaces_cnt:,} patches</div>
                    <div><strong>Feature Dimension:</strong> {feature_dim}-d (HOG + HSV + Texture)</div>
                    <div><strong>Inference Speed:</strong> {latency:.3f} ms / space</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Comparative Test Metrics
        t_clean = m.get("test_clean", {})
        t_raw = m.get("test_all_labels", {})

        col_clean, col_raw = st.columns(2, gap="medium")

        with col_clean:
            err_clean = t_clean.get("occupancy_count", {}).get("mean_abs_count_error", 0.06)
            st.markdown(
                f"""
                <div class="ps-report-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h4 style="margin:0; font-size:1rem; font-weight:700;">Audited Clean Test Set</h4>
                        <span class="ps-pill ps-pill-success">Cleaned Benchmark</span>
                    </div>
                    <p style="font-size:0.8rem; opacity:0.75; margin:4px 0 0 0;">Excludes 2 known mislabeled camera images</p>
                    <div class="ps-report-grid">
                        <div class="ps-mini-stat">
                            <div class="ps-mini-label">Accuracy</div>
                            <div class="ps-mini-val" style="color:#10b981;">{t_clean.get('accuracy', 0.9989)*100:.2f}%</div>
                        </div>
                        <div class="ps-mini-stat">
                            <div class="ps-mini-label">F1-Score</div>
                            <div class="ps-mini-val" style="color:#0ea5e9;">{t_clean.get('f1', 0.9988):.4f}</div>
                        </div>
                        <div class="ps-mini-stat">
                            <div class="ps-mini-label">Precision</div>
                            <div class="ps-mini-val">{t_clean.get('precision', 0.9988)*100:.2f}%</div>
                        </div>
                        <div class="ps-mini-stat">
                            <div class="ps-mini-label">Recall</div>
                            <div class="ps-mini-val">{t_clean.get('recall', 0.9988)*100:.2f}%</div>
                        </div>
                    </div>
                    <div style="margin-top:12px; font-size:0.8rem; opacity:0.8;">
                        <strong>Mean Count Error:</strong> {err_clean:.2f} spaces / image
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_raw:
            err_raw = t_raw.get("occupancy_count", {}).get("mean_abs_count_error", 2.06)
            st.markdown(
                f"""
                <div class="ps-report-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h4 style="margin:0; font-size:1rem; font-weight:700;">Raw Test Set (All Labels)</h4>
                        <span class="ps-pill" style="background:var(--ps-warning-bg); color:var(--ps-warning);">Original Dataset</span>
                    </div>
                    <p style="font-size:0.8rem; opacity:0.75; margin:4px 0 0 0;">Includes raw dataset label discrepancies</p>
                    <div class="ps-report-grid">
                        <div class="ps-mini-stat">
                            <div class="ps-mini-label">Accuracy</div>
                            <div class="ps-mini-val" style="color:#f59e0b;">{t_raw.get('accuracy', 0.9632)*100:.2f}%</div>
                        </div>
                        <div class="ps-mini-stat">
                            <div class="ps-mini-label">F1-Score</div>
                            <div class="ps-mini-val" style="color:#0ea5e9;">{t_raw.get('f1', 0.9616):.4f}</div>
                        </div>
                        <div class="ps-mini-stat">
                            <div class="ps-mini-label">Precision</div>
                            <div class="ps-mini-val">{t_raw.get('precision', 0.9271)*100:.2f}%</div>
                        </div>
                        <div class="ps-mini-stat">
                            <div class="ps-mini-label">Recall</div>
                            <div class="ps-mini-val">{t_raw.get('recall', 0.9988)*100:.2f}%</div>
                        </div>
                    </div>
                    <div style="margin-top:12px; font-size:0.8rem; opacity:0.8;">
                        <strong>Mean Count Error:</strong> {err_raw:.2f} spaces / image
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Validation Comparison Table & Confusion Matrix
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        c_val, c_cm = st.columns([1.1, 0.9], gap="medium")

        with c_val:
            st.markdown(
                """
                <div class="ps-card" style="height: 100%;">
                    <div class="ps-section-header">
                        <div class="ps-section-title">
                            <span>📈 Validation Model Benchmark</span>
                        </div>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            val_data = m.get("validation", {})
            if val_data:
                rows = []
                for model_key, stats in val_data.items():
                    is_selected = model_key == selected_model_name
                    rows.append({
                        "Candidate Model": f"{'🏆 ' if is_selected else ''}{model_key}",
                        "Accuracy": f"{stats.get('accuracy', 0)*100:.2f}%",
                        "Precision": f"{stats.get('precision', 0)*100:.2f}%",
                        "Recall": f"{stats.get('recall', 0)*100:.2f}%",
                        "F1-Score": f"{stats.get('f1', 0):.4f}",
                        "Fit Time": f"{stats.get('fit_seconds', 0):.1f}s",
                    })
                df_val = pd.DataFrame(rows)
                st.dataframe(df_val, use_container_width=True, hide_index=True)

            st.markdown("</div>", unsafe_allow_html=True)

        with c_cm:
            st.markdown(
                """
                <div class="ps-card" style="height: 100%;">
                    <div class="ps-section-header">
                        <div class="ps-section-title">
                            <span>🎯 Confusion Matrix (Clean Test Set)</span>
                        </div>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
            if os.path.exists(cm_path):
                st.image(cm_path, use_container_width=True)
            else:
                st.info("Confusion matrix image not found in reports directory.")
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("⚠️ Metrics report not found. Run `python train_model.py` to generate the model report.")
