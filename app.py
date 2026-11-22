"""
ReviewLens — review sentiment intelligence dashboard.

The app uses the BERT model and optional TF-IDF baseline from the supplied
starter project. New interface and workflows: single review analysis,
CSV batch analysis, charts, downloadable results, and session history.
"""
import io
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from pypdf import PdfReader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import LABELS, METRICS_FILE, MODELS_DIR, clean_text

APP_NAME = "ReviewLens"
FALLBACK = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LEN, STRIDE, MAX_WINDOWS = 256, 32, 32

st.set_page_config(
    page_title=f"{APP_NAME} | Sentiment Intelligence",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0b1020; color: #eef2ff; }
[data-testid="stSidebar"] { background: #10182b; border-right: 1px solid #26334e; }
[data-testid="stHeader"] { background: rgba(11,16,32,0.92); }
.block-container { padding-top: 1.7rem; max-width: 1450px; }
h1,h2,h3 { font-family: 'Manrope', sans-serif !important; color: #f5f7ff !important; }
p, label, .stCaption { color: #b8c3d9; }
.hero {
  padding: 1.5rem 1.7rem; border-radius: 22px;
  background: linear-gradient(115deg,#17284a 0%,#25245a 55%,#3b2460 100%);
  border: 1px solid #39446d; margin-bottom: 1.2rem;
}
.hero-kicker { color:#a5b4fc; font-size:.76rem; letter-spacing:.16em; font-weight:700; }
.hero-title { color:#fff; font-family:Manrope,sans-serif; font-size:2.25rem; font-weight:800; margin:.35rem 0; }
.hero-copy { color:#d2d9ee; max-width:760px; }
div[data-testid="stMetric"] {
  background:#131e33; border:1px solid #293957; padding:15px 17px;
  border-radius:16px;
}
div[data-testid="stMetricLabel"] p { color:#aebbd3 !important; }
div[data-testid="stMetricValue"] { color:#f5f7ff !important; }
.stButton>button {
  border-radius:11px; border:1px solid #6475d9; background:#5968d9;
  color:white; font-weight:700; min-height:2.7rem;
}
.stButton>button:hover { background:#7481ed; border-color:#9aa4ff; color:white; }
.stTextArea textarea, .stTextInput input {
  background:#111b2e !important; color:#f5f7ff !important;
  border:1px solid #344461 !important; border-radius:12px !important;
}
[data-testid="stFileUploader"] { background:#111b2e; border:1px dashed #465a7b; border-radius:14px; padding:10px; }
div[data-testid="stDataFrame"] { border:1px solid #293957; border-radius:12px; overflow:hidden; }
hr { border-color:#263653; }
.small-note { color:#91a0ba; font-size:.83rem; }
.pill { display:inline-block; padding:5px 10px; border-radius:999px; background:#1d2d4a; color:#c7d2fe; font-size:.78rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading language model…")
def load_bert():
    local = MODELS_DIR / "bert"
    path = str(local) if local.exists() else FALLBACK
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    return tok, model, local.exists()

@st.cache_resource
def load_baseline():
    p = MODELS_DIR / "baseline.joblib"
    return joblib.load(p) if p.exists() else None

def bert_predict(text: str):
    tok, model, _ = load_bert()
    enc = tok(text, truncation=True, max_length=MAX_LEN, stride=STRIDE,
              return_overflowing_tokens=True, padding=True, return_tensors="pt")
    enc.pop("overflow_to_sample_mapping", None)
    enc = {k: v[:MAX_WINDOWS] for k, v in enc.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1]
    return float(probs.mean().item()), probs.tolist()

def verdict(p_pos: float, neutral_band: float):
    if abs(p_pos - 0.5) < neutral_band:
        return "Neutral / Mixed", 1 - abs(p_pos - 0.5) * 2
    return ("Positive", p_pos) if p_pos > 0.5 else ("Negative", 1 - p_pos)

def extract_file(upload):
    name = upload.name.lower()
    if name.endswith(".pdf"):
        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(upload.getvalue())).pages)
    return upload.getvalue().decode("utf-8", errors="ignore")

def analyze_one(text, neutral_band):
    cleaned = clean_text(str(text))
    if len(cleaned.split()) < 3:
        raise ValueError("Please provide at least three words.")
    p, windows = bert_predict(cleaned)
    label, confidence = verdict(p, neutral_band)
    return {
        "text": cleaned, "sentiment": label, "confidence": round(confidence * 100, 2),
        "positive_probability": round(p * 100, 2),
        "word_count": len(cleaned.split()), "windows": len(windows),
    }

def sentiment_color(label):
    return {"Positive":"#34d399", "Negative":"#fb7185", "Neutral / Mixed":"#fbbf24"}.get(label, "#a5b4fc")

if "history" not in st.session_state:
    st.session_state.history = []
if "batch_result" not in st.session_state:
    st.session_state.batch_result = None

with st.sidebar:
    st.markdown("## 🔎 ReviewLens")
    st.caption("SENTIMENT INTELLIGENCE")
    page = st.radio("WORKSPACE", ["Overview", "Analyze text", "Batch review lab", "Model evaluation", "History"], label_visibility="visible")
    st.divider()
    neutral_band = st.slider("Neutral sensitivity", 0.00, 0.30, 0.10, 0.01,
        help="Predictions close to 50% positive probability are labelled Neutral / Mixed.")
    st.caption("Model settings")
    st.caption("BERT window length: 256 tokens")
    st.caption("Maximum windows: 32")
    st.divider()
    st.markdown('<div class="small-note">Predictions are model estimates, not objective facts. Results can vary by domain, sarcasm and context.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-kicker">TEXT ANALYTICS WORKSPACE</div>
  <div class="hero-title">Understand the voice behind every review.</div>
  <div class="hero-copy">Explore sentiment in individual comments or entire review datasets. Turn unstructured feedback into clear, readable signals.</div>
</div>
""", unsafe_allow_html=True)

# Load status is shown without blocking the overview page.
if page == "Overview":
    st.subheader("Your analysis workspace")
    try:
        _, _, is_local = load_bert()
        if is_local:
            st.success("Fine-tuned local BERT model detected.")
        else:
            st.info(f"Using the public pretrained model ({FALLBACK}). Train your own model to replace it.")
    except Exception as e:
        st.error(f"Model could not be loaded: {e}")
    history = pd.DataFrame(st.session_state.history)
    total = len(history)
    positive = int((history["sentiment"] == "Positive").sum()) if total else 0
    negative = int((history["sentiment"] == "Negative").sum()) if total else 0
    mixed = total - positive - negative
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Reviews analyzed", total)
    c2.metric("Positive", positive)
    c3.metric("Negative", negative)
    c4.metric("Neutral / Mixed", mixed)
    st.markdown("### Get started")
    a,b = st.columns(2)
    with a:
        st.markdown("#### ✍️ Analyze one review")
        st.write("Paste text or upload a TXT/PDF document to receive a sentiment prediction and confidence estimate.")
        if st.button("Open text analyzer", use_container_width=True):
            st.session_state["nav_to"] = "Analyze text"
            st.rerun()
    with b:
        st.markdown("#### 📊 Explore a review dataset")
        st.write("Upload a CSV containing a review column, analyze each row, and export the enriched dataset.")
        if st.button("Open batch review lab", use_container_width=True):
            st.session_state["nav_to"] = "Batch review lab"
            st.rerun()
    st.caption("Tip: Start with a short review before processing a large CSV. The first model load may take a few minutes.")

elif page == "Analyze text":
    st.subheader("Single text analysis")
    st.write("Paste a customer review, feedback message, or document. You can also upload a TXT or PDF file.")
    uploaded = st.file_uploader("Choose a document", type=["txt","pdf"], key="single_upload")
    initial = extract_file(uploaded) if uploaded else ""
    text = st.text_area("Text to analyze", value=initial, height=220, placeholder="Example: The delivery was quick and the product quality was excellent…")
    if st.button("Analyze sentiment", type="primary"):
        try:
            with st.spinner("Reading context and estimating sentiment…"):
                result = analyze_one(text, neutral_band)
            st.session_state.history.insert(0, {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "sentiment": result["sentiment"], "confidence": result["confidence"],
                "word_count": result["word_count"], "text": result["text"][:240]
            })
            st.markdown("### Analysis result")
            r1,r2,r3 = st.columns(3)
            r1.metric("Predicted sentiment", result["sentiment"])
            r2.metric("Confidence estimate", f'{result["confidence"]:.1f}%')
            r3.metric("Words processed", result["word_count"])
            st.progress(result["positive_probability"]/100, text=f'Positive probability: {result["positive_probability"]:.1f}%')
            st.caption(f'{result["windows"]} text window(s) evaluated. Confidence is the model score associated with the selected label or neutral heuristic.')
            with st.expander("View cleaned text"):
                st.write(result["text"])
            base = load_baseline()
            if base is not None:
                p_base = float(base.predict_proba([result["text"]])[0][1])
                base_label, base_conf = verdict(p_base, neutral_band)
                st.markdown("#### Model cross-check")
                st.dataframe(pd.DataFrame([
                    {"Model":"BERT", "Prediction":result["sentiment"], "Confidence":f'{result["confidence"]:.1f}%'},
                    {"Model":"TF-IDF + Logistic Regression", "Prediction":base_label, "Confidence":f'{base_conf*100:.1f}%'}
                ]), hide_index=True, use_container_width=True)
        except Exception as e:
            st.error(str(e))

elif page == "Batch review lab":
    st.subheader("Batch review lab")
    st.write("Upload a CSV file with a column containing review text. The app adds sentiment and confidence columns.")
    file = st.file_uploader("Upload review dataset", type=["csv"], key="batch_csv")
    if file:
        try:
            data = pd.read_csv(file)
            st.write("Dataset preview")
            st.dataframe(data.head(8), use_container_width=True, hide_index=True)
            text_col = st.selectbox("Select the review text column", data.columns.tolist())
            limit = st.number_input("Maximum rows to analyze", min_value=1, max_value=min(500, len(data)), value=min(50, len(data)), step=1)
            if st.button("Analyze dataset", type="primary"):
                rows = data.head(int(limit)).copy()
                results = []
                progress = st.progress(0, text="Starting batch analysis…")
                for i, value in enumerate(rows[text_col].fillna("").astype(str)):
                    try:
                        res = analyze_one(value, neutral_band)
                        results.append({k:res[k] for k in ["sentiment","confidence","positive_probability","word_count"]})
                    except Exception:
                        results.append({"sentiment":"Not analyzed","confidence":0.0,"positive_probability":0.0,"word_count":0})
                    progress.progress((i+1)/len(rows), text=f"Analyzed {i+1} of {len(rows)}")
                for col in results[0].keys() if results else []:
                    rows[col] = [r[col] for r in results]
                st.session_state.batch_result = rows
                for _, r in rows.iterrows():
                    if r["sentiment"] != "Not analyzed":
                        st.session_state.history.insert(0, {
                            "time":datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "sentiment":r["sentiment"], "confidence":r["confidence"],
                            "word_count":r["word_count"], "text":str(r[text_col])[:240]
                        })
                st.success("Batch analysis complete.")
        except Exception as e:
            st.error(f"Could not read this CSV: {e}")
    if st.session_state.batch_result is not None:
        result = st.session_state.batch_result
        st.markdown("### Batch results")
        counts = result["sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Reviews")
        c1,c2 = st.columns([1,1])
        with c1:
            st.dataframe(result, use_container_width=True, hide_index=True)
        with c2:
            fig = px.pie(counts, names="Sentiment", values="Reviews", hole=.58,
                         title="Sentiment breakdown", color="Sentiment",
                         color_discrete_map={"Positive":"#34d399","Negative":"#fb7185","Neutral / Mixed":"#fbbf24","Not analyzed":"#94a3b8"})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8edff")
            st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download analyzed CSV", result.to_csv(index=False).encode("utf-8"), "reviewlens_results.csv", "text/csv")
        st.download_button("Download summary JSON", json.dumps(counts.to_dict(orient="records"), indent=2).encode("utf-8"), "reviewlens_summary.json", "application/json")

elif page == "Model evaluation":
    st.subheader("Model evaluation")
    if METRICS_FILE.exists():
        metrics = json.loads(METRICS_FILE.read_text())
        rows = []
        for name, values in metrics.items():
            rows.append({"Model":name, **{k:round(values[k],4) for k in ("accuracy","precision","recall","f1") if k in values},
                         "Training samples":values.get("train_samples"), "Training time (s)":values.get("train_seconds")})
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        metric_cols = [c for c in ["accuracy","precision","recall","f1"] if c in df.columns]
        if metric_cols:
            long = df.melt(id_vars="Model", value_vars=metric_cols, var_name="Metric", value_name="Score")
            fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group", range_y=[0,1], title="Model performance comparison")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8edff")
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Metrics are displayed from models/metrics.json. Run the training scripts to generate or refresh them.")
    else:
        st.info("No evaluation metrics found yet. Train the baseline and/or BERT model, then revisit this page.")
    st.markdown("#### Training commands")
    st.code("python train_baseline.py\npython train_bert.py --train_samples 5000 --epochs 2", language="bash")

elif page == "History":
    st.subheader("Session history")
    st.write("Recent analyses from this browser session. History is cleared when the Streamlit session resets.")
    if st.session_state.history:
        hist = pd.DataFrame(st.session_state.history)
        st.dataframe(hist, use_container_width=True, hide_index=True)
        st.download_button("Export session history", hist.to_csv(index=False).encode("utf-8"), "reviewlens_history.csv", "text/csv")
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No reviews analyzed in this session yet.")
