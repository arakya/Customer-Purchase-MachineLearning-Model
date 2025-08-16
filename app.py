import streamlit as st
import numpy as np
import pickle
from pathlib import Path

BASE = Path(__file__).parent

@st.cache_resource
def load_artifacts():
    with open(BASE / "stacked_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# -------------------------
# Page config + lightweight theming
# -------------------------
st.set_page_config(
    page_title="Customer Purchase Prediction",
    page_icon="🛒",
    layout="centered"
)

# Custom CSS for a colorful, card-like vibe (no plain white)
st.markdown("""
<style>
/* gradient background */
.stApp {
    background: linear-gradient(135deg, #1f2937 0%, #3b82f6 40%, #22c55e 100%);
    background-attachment: fixed;
}

/* translucent card containers */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.card {
    background: rgba(255,255,255,0.10);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 16px;
    padding: 1.25rem 1.25rem 0.75rem 1.25rem;
    box-shadow: 0 6px 30px rgba(0,0,0,0.20);
}

/* headings */
h1, h2, h3, h4 {
    color: #f8fafc !important;
}

/* labels and text */
label, .stRadio > label, .stSelectbox > label, .stNumberInput > label, .stMarkdown {
    color: #e5e7eb !important;
}

/* buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    border: none;
    padding: 0.6rem 1.1rem;
}

/* success + warning pills */
.result-pill {
    display: inline-block;
    padding: 0.35rem 0.65rem;
    border-radius: 999px;
    font-weight: 700;
    letter-spacing: .2px;
}

.result-buy { background: #22c55e; color: #052e16; }
.result-nobuy { background: #f59e0b; color: #3b1d00; }

/* metrics */
[data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
    color: #f8fafc !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Load artifacts
# -------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    here = Path(__file__).parent
    model_path = here / "stacked_model.pkl"
    scaler_path = here / "scaler.pkl"

    if not model_path.exists():
        st.stop()  # hard stop with Streamlit error UI below

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    scaler = None
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Failed to load model/scaler: {e}")
    st.stop()

# -------------------------
# Header
# -------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("# 🛒 Customer Purchase Prediction")
st.markdown(
    "Predict whether a customer will make a purchase based on their profile and interaction stats.",
)

st.markdown("</div>", unsafe_allow_html=True)
st.write("")

# -------------------------
# Inputs
# -------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Inputs")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    annual_income = st.number_input("Annual Income (₹)", min_value=0, max_value=1_000_000, value=120_000, step=1000)
    number_of_purchases = st.number_input("Number of Purchases", min_value=0, max_value=1000, value=3, step=1)

with col2:
    # If your dataset used coded categories (1..N), keep that mapping here.
    product_category = st.number_input("Product Category (code)", min_value=1, max_value=10, value=1, step=1)
    time_spent_on_website = st.number_input("Time Spent on Website (minutes)", min_value=0, max_value=10_000, value=15, step=1)
    loyalty_program = st.selectbox("Loyalty Program", ["No", "Yes"])
    discounts_availed = st.number_input("Discounts Availed", min_value=0, max_value=100, value=1, step=1)

st.markdown("</div>", unsafe_allow_html=True)
st.write("")

# -------------------------
# Prepare features (order MUST match training)
# ["age","gender","annual_income","number_of_purchases","product_category",
#  "time_spent_on_website","loyalty_program","discounts_availed"]
# -------------------------
gender_num = 1 if gender.lower().startswith("m") else 0
loyalty_num = 1 if loyalty_program == "Yes" else 0

raw_features = np.array([[
    age,
    gender_num,
    annual_income,
    number_of_purchases,
    product_category,
    time_spent_on_website,
    loyalty_num,
    discounts_availed
]], dtype=float)

# Sanity checks mirroring your older validators (tweak if needed)
def validate():
    if not (10_000 <= annual_income <= 500_000):
        st.error("Annual income must be between 10,000 and 500,000.")
        return False
    return True

# -------------------------
# Predict
# -------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
left, right = st.columns([1,1])

with left:
    go = st.button("Predict", type="primary", use_container_width=True)

with right:
    reset = st.button("Reset inputs", use_container_width=True)
    if reset:
        st.experimental_rerun()

if go:
    if not validate():
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    X = raw_features
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            st.warning(f"Scaler transform failed, using raw features. Details: {e}")

    try:
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
        y_pred = model.predict(X)
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    pred = int(y_pred[0])
    prob_buy = float(proba[0, 1]) if proba is not None else None

    # Display result
    st.write("")
    if pred == 1:
        st.markdown('<span class="result-pill result-buy">LIKELY TO PURCHASE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="result-pill result-nobuy">UNLIKELY TO PURCHASE</span>', unsafe_allow_html=True)

    # Nice metrics row
    m1, m2 = st.columns(2)
    m1.metric("Prediction", "Buy" if pred == 1 else "No Buy")
    if prob_buy is not None:
        m2.metric("Confidence (P=Buy)", f"{prob_buy*100:0.1f}%")

    with st.expander("View input feature vector (ordered as in training)"):
        st.write(
            {
                "age": age,
                "gender (1=Male)": gender_num,
                "annual_income": annual_income,
                "number_of_purchases": number_of_purchases,
                "product_category": product_category,
                "time_spent_on_website": time_spent_on_website,
                "loyalty_program (1=Yes)": loyalty_num,
                "discounts_availed": discounts_availed,
            }
        )

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Footer note (kept minimal)
# -------------------------
st.markdown(
    """
    <div style="text-align:center; color:#e5e7eb; margin-top:1rem;">
        Built with Streamlit • Model: stacked_model.pkl • Scaler applied if available
    </div>
    """,
    unsafe_allow_html=True,
)
