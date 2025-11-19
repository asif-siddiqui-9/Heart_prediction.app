import streamlit as st
import pandas as pd
import joblib
import time
from datetime import datetime

# Optional PDF support
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

import plotly.graph_objects as go

# Constants
AGE_MIN, AGE_MAX, AGE_DEFAULT = 18, 100, 40
BP_MIN, BP_MAX, BP_DEFAULT = 80, 200, 120
CHOL_MIN, CHOL_MAX, CHOL_DEFAULT = 100, 600, 200
HR_MIN, HR_MAX, HR_DEFAULT = 60, 220, 150

# ------------------------ PAGE CONFIG ------------------------ #
st.set_page_config(
    page_title="Heart Stroke Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

# ------------------------ SESSION STATE ------------------------ #
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# ------------------------ CUSTOM CSS ------------------------ #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Background */
.stApp {
    background: radial-gradient(circle at top left, #0a0e27 0%, #020617 35%, #0f172a 100%);
    color: #f1f5f9;
}

/* Fade-in animation */
@keyframes fadeInUp {
    from { opacity: 0; transform: translate3d(0, 20px, 0); }
    to { opacity: 1; transform: translate3d(0, 0, 0); }
}

/* Glow animation for hero ring */
@keyframes glowRing {
    0% { box-shadow: 0 0 0 0 rgba(248, 113, 113, 0.3); }
    50% { box-shadow: 0 0 20px 15px rgba(248, 113, 113, 0.08); }
    100% { box-shadow: 0 0 0 0 rgba(248, 113, 113, 0.0); }
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}

/* Hero section */
.hero-wrapper {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 2rem;
    padding: 1.5rem 0 0.5rem 0;
    animation: fadeInUp 0.6s ease-out;
}
.hero-text {
    max-width: 60%;
}
.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.8rem;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(56,189,248,0.25), rgba(168,85,247,0.3));
    border: 1px solid rgba(125,211,252,0.8);
    color: #bae6fd;
    box-shadow: 0 4px 12px rgba(56,189,248,0.2);
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.15;
    margin-top: 0.8rem;
    background: linear-gradient(135deg, #f97316, #fb7185, #a855f7, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-subtitle {
    font-size: 1rem;
    color: #cbd5e1;
    margin-top: 0.6rem;
    line-height: 1.6;
}
.hero-highlight {
    color: #f97316;
    font-weight: 700;
}

/* Hero right side */
.hero-right {
    flex: 1;
    display: flex;
    justify-content: center;
}
.heart-orbit {
    position: relative;
    width: 220px;
    height: 220px;
    border-radius: 999px;
    background: radial-gradient(circle at 30% 20%, #f97316, transparent 60%),
                radial-gradient(circle at 70% 80%, #ec4899, transparent 55%);
    animation: glowRing 2.8s infinite ease-out;
    display: flex;
    align-items: center;
    justify-content: center;
}
.heart-inner {
    width: 160px;
    height: 160px;
    border-radius: 999px;
    background: rgba(10,14,39,0.98);
    border: 1px solid rgba(248,250,252,0.2);
    box-shadow: 0 20px 50px rgba(0,0,0,0.9);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    animation: pulse 3s infinite ease-in-out;
}
.heart-emoji {
    font-size: 2.8rem;
}
.heart-label {
    margin-top: 0.4rem;
    font-size: 0.85rem;
    color: #f1f5f9;
    font-weight: 600;
}
.heart-pulse {
    font-size: 0.72rem;
    color: #22c55e;
    margin-top: 0.2rem;
}

/* Section cards */
.section-card {
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 1.2rem;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    transition: all 0.3s ease;
}
.section-card:hover {
    border-color: rgba(148, 163, 184, 0.4);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}

/* Section headers */
.section-header {
    font-weight: 700;
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-header span.icon {
    font-size: 1.3rem;
}
.section-header span.text-gradient-pink-green {
    background: linear-gradient(90deg,#fb7185,#22c55e);
    -webkit-background-clip:text;
    -webkit-text-fill-color: transparent;
}
.section-header span.text-gradient-blue-pink {
    background: linear-gradient(90deg,#38bdf8,#ec4899);
    -webkit-background-clip:text;
    -webkit-text-fill-color: transparent;
}
.section-header span.text-gradient-pink-blue {
    background: linear-gradient(90deg,#fb7185,#38bdf8);
    -webkit-background-clip:text;
    -webkit-text-fill-color: transparent;
}
.section-header span.text-gradient-green-orange {
    background: linear-gradient(90deg,#22c55e,#f97316);
    -webkit-background-clip:text;
    -webkit-text-fill-color: transparent;
}

/* Gradient dividers */
.gradient-line-pink-green {
    border:0;
    height:2px;
    background:linear-gradient(90deg,#fb7185,#22c55e,#38bdf8);
    opacity:0.8;
    margin-bottom:0.8rem;
}
.gradient-line-blue-pink {
    border:0;
    height:2px;
    background:linear-gradient(90deg,#38bdf8,#ec4899,#8b5cf6);
    opacity:0.8;
    margin-bottom:0.8rem;
}
.gradient-line-pink-blue {
    border:0;
    height:2px;
    background:linear-gradient(90deg,#fb7185,#38bdf8,#22c55e);
    opacity:0.8;
    margin-bottom:0.8rem;
}

/* Input styling */
.stSlider label, .stSelectbox label, .stNumberInput label {
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    color: #e2e8f0 !important;
}

.stSlider > div > div, .stSelectbox > div, .stNumberInput > div > div {
    background: rgba(15, 23, 42, 0.85) !important;
    border-radius: 0.75rem !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    transition: all 0.3s ease !important;
}

.stSlider > div > div:hover, .stSelectbox > div:hover, .stNumberInput > div > div:hover {
    border-color: rgba(148, 163, 184, 0.4) !important;
    transform: translateY(-1px);
}

/* Button */
.stButton>button {
    width: 100%;
    border-radius: 999px;
    border: 0;
    padding: 1rem 1.5rem;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.05em;
    background: linear-gradient(135deg, #f97316, #ec4899, #8b5cf6);
    background-size: 200% 200%;
    color: white;
    box-shadow: 0 20px 45px rgba(236, 72, 153, 0.4);
    transition: all 0.3s ease-in-out;
    animation: gradientShift 3s ease infinite;
}
.stButton>button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 25px 55px rgba(236, 72, 153, 0.6);
}

@keyframes gradientShift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

/* Metric cards */
.stMetric {
    background: rgba(15, 23, 42, 0.9);
    padding: 1rem 1.2rem;
    border-radius: 1.2rem;
    border: 1px solid rgba(148, 163, 184, 0.3);
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

/* Risk badge */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    font-size: 0.9rem;
    font-weight: 700;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.risk-badge.low {
    background: rgba(22,163,74,0.2);
    color: #4ade80;
    border: 1px solid rgba(74,222,128,0.6);
}
.risk-badge.moderate {
    background: rgba(234,179,8,0.2);
    color: #facc15;
    border: 1px solid rgba(250,204,21,0.6);
}
.risk-badge.high {
    background: rgba(239,68,68,0.2);
    color: #fca5a5;
    border: 1px solid rgba(248,113,113,0.7);
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 3rem;
    padding: 2rem 0 2.5rem 0;
    border-top: 2px solid transparent;
    border-image: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.3), transparent) 1;
}

.footer-title {
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.5rem;
    background: linear-gradient(90deg, #38bdf8, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.footer-subtitle {
    font-size: 0.88rem;
    color: #cbd5e1;
    margin-bottom: 1.5rem;
    line-height: 1.6;
}

.social-buttons {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin: 1.5rem 0;
    flex-wrap: wrap;
}

.social-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.75rem 1.5rem;
    border-radius: 999px;
    font-size: 0.95rem;
    font-weight: 600;
    text-decoration: none;
    color: white;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

.social-btn:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
}

.btn-github {
    background: linear-gradient(135deg, #24292e, #4a5568);
}

.btn-linkedin {
    background: linear-gradient(135deg, #0077b5, #00a0dc);
}

.btn-email {
    background: linear-gradient(135deg, #ea4335, #f97316);
}

.footer-copyright {
    margin-top: 1.5rem;
    font-size: 0.8rem;
    color: #64748b;
}

/* Responsive */
@media (max-width: 768px) {
    .hero-title {
        font-size: 2rem !important;
    }
    .hero-wrapper {
        flex-direction: column;
    }
    .hero-text {
        max-width: 100%;
    }
    .footer-links {
        flex-direction: column;
    }
}
</style>
""", unsafe_allow_html=True)

# ------------------------ LOAD ARTIFACTS ------------------------ #
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("Heart_LR.pkl")
        scaler = joblib.load("Heart_scaler.pkl")
        expected_columns = joblib.load("Heart_column.pkl")
        return model, scaler, expected_columns
    except FileNotFoundError as e:
        st.error("❌ Model files not found. Please ensure Heart_LR.pkl, Heart_scaler.pkl, and Heart_column.pkl are in the same directory.")
        st.info("📁 Missing file: " + str(e))
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model files: {str(e)}")
        st.stop()

model, scaler, expected_columns = load_artifacts()
expected_columns = list(expected_columns)

# ------------------------ VALIDATION FUNCTIONS ------------------------ #
def validate_inputs(age, bp, chol, hr):
    """Validate user inputs and show warnings if needed"""
    warnings = []
    
    if bp < 90 or bp > 180:
        warnings.append("⚠️ Blood pressure seems unusual. Please verify your reading.")
    
    if chol > 300:
        warnings.append("⚠️ Very high cholesterol detected. Please consult a doctor.")
    
    if hr < 50:
        warnings.append("⚠️ Low heart rate detected. This may need medical attention.")
    elif hr > 200:
        warnings.append("⚠️ Extremely high heart rate. Please verify.")
    
    return warnings

# ------------------------ HERO SECTION ------------------------ #
hero_col1, hero_col2 = st.columns([1.7, 1.1])

with hero_col1:
    st.markdown("""
    <div class="hero-wrapper">
      <div class="hero-text">
        <div class="hero-pill">
          <span>🧪 AI-Powered Screening</span> • <span>Not a medical diagnosis</span>
        </div>
        <div class="hero-title">
          Heart Stroke Risk<br/>Prediction Dashboard
        </div>
        <div class="hero-subtitle">
          Built with ❤️ by <span class="hero-highlight">ASIF SIDDIQUE</span>. 
          Enter your health details to get an AI-based estimation of your
          <b>heart disease risk</b> and quick lifestyle suggestions.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with hero_col2:
    st.markdown("""
    <div class="hero-wrapper" style="justify-content:flex-end;">
      <div class="hero-right">
        <div class="heart-orbit">
          <div class="heart-inner">
            <div class="heart-emoji">❤️</div>
            <div class="heart-label">Your Heart Snapshot</div>
            <div class="heart-pulse">Live AI Risk Check</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# ------------------------ MAIN LAYOUT ------------------------ #
left_col, right_col = st.columns([1.5, 1])

# -------- LEFT: INPUTS -------- #
with left_col:
    st.markdown(
        """
        <div class="section-header">
            <span class="icon">🧍‍♂️</span>
            <span class="text-gradient-pink-green">Personal & Vital Information</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="gradient-line-pink-green">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age = st.slider(
            "Age", 
            AGE_MIN, AGE_MAX, AGE_DEFAULT,
            help="Your current age in years"
        )
        sex = st.selectbox(
            "Sex", 
            ["M", "F"],
            help="Biological sex assigned at birth"
        )
        max_hr = st.slider(
            "Max Heart Rate", 
            HR_MIN, HR_MAX, HR_DEFAULT,
            help="Maximum heart rate achieved during exercise (bpm)"
        )
    with c2:
        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)", 
            BP_MIN, BP_MAX, BP_DEFAULT,
            help="Normal: 90-120 | Elevated: 120-129 | High: ≥130"
        )
        cholesterol = st.number_input(
            "Cholesterol (mg/dL)", 
            CHOL_MIN, CHOL_MAX, CHOL_DEFAULT,
            help="Normal: <200 | Borderline: 200-239 | High: ≥240"
        )
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL", 
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Whether your fasting blood sugar is above 120 mg/dL"
        )

    st.markdown(
        """
        <div class="section-header" style="margin-top:1.5rem;">
            <span class="icon">🫀</span>
            <span class="text-gradient-blue-pink">Heart Condition Details</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="gradient-line-blue-pink">', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        chest_pain = st.selectbox(
            "Chest Pain Type", 
            ["ATA", "NAP", "TA", "ASY"],
            help="ATA: Atypical Angina | NAP: Non-Anginal Pain | TA: Typical Angina | ASY: Asymptomatic"
        )
        resting_ecg = st.selectbox(
            "Resting ECG", 
            ["Normal", "ST", "LVH"],
            help="Normal | ST: ST-T wave abnormality | LVH: Left ventricular hypertrophy"
        )
    with c4:
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina", 
            ["Y", "N"],
            format_func=lambda x: "Yes" if x == "Y" else "No",
            help="Chest pain triggered by physical activity"
        )
        st_slope = st.selectbox(
            "ST Slope", 
            ["Up", "Flat", "Down"],
            help="Slope of peak exercise ST segment"
        )

    oldpeak = st.slider(
        "Oldpeak (ST Depression)", 
        0.0, 6.0, 1.0, step=0.1,
        help="ST depression induced by exercise relative to rest"
    )

    st.markdown("")
    
    # Show validation warnings
    warnings = validate_inputs(age, resting_bp, cholesterol, max_hr)
    if warnings:
        for warning in warnings:
            st.warning(warning)
    
    predict_btn = st.button("🔍 ANALYZE HEART STROKE RISK")

# -------- RIGHT: SUMMARY & TIPS -------- #
with right_col:
    st.markdown(
        """
        <div class="section-header">
            <span class="icon">📊</span>
            <span class="text-gradient-pink-blue">Live Risk Summary</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="gradient-line-pink-blue">', unsafe_allow_html=True)

    st.markdown(
        "After you click **ANALYZE HEART STROKE RISK**, your details will be processed by the "
        "trained machine learning model, and an estimated risk level will appear below."
    )

    placeholder_result = st.empty()
    placeholder_metrics = st.empty()
    placeholder_gauge = st.empty()
    placeholder_health = st.empty()
    placeholder_download = st.empty()

    st.markdown("---")

    st.markdown(
        """
        <div class="section-header" style="margin-top:1rem;">
            <span class="icon">💡</span>
            <span class="text-gradient-green-orange">Heart Health Micro-Tips</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
- 🚶‍♂️ Aim for at least **30 minutes of movement** on most days.  
- 🥗 Prefer **whole grains, fruits, vegetables, and lean proteins** over processed foods.  
- 🌙 Keep a regular sleep routine of **7–8 hours** per night.  
- 🧘‍♀️ Use deep-breathing or meditation to **manage stress**.  
- 🚭 Avoid **smoking** and limit **alcohol** intake.

> This app is for **educational support only** and is **not a medical diagnosis**.  
> Always consult a qualified doctor for clinical decisions.
        """
    )

# ------------------------ HELPER: RISK CATEGORY ------------------------ #
def get_risk_category(risk_score, prediction):
    """Returns (label, class_name, emoji)"""
    if risk_score is not None:
        if risk_score < 20:
            return "Low Risk", "low", "🟢"
        elif risk_score < 50:
            return "Moderate Risk", "moderate", "🟡"
        else:
            return "High Risk", "high", "🔴"
    else:
        if prediction == 1:
            return "High Risk", "high", "🔴"
        else:
            return "Low Risk", "low", "🟢"

# ------------------------ PREDICTION LOGIC ------------------------ #
if predict_btn:
    # Build raw input
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    # Add missing expected columns as 0
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Keep only expected columns in correct order
    input_df = input_df[expected_columns]

    with st.spinner("🔄 Running AI model on your inputs..."):
        progress = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Preprocessing data...")
        progress.progress(25)
        time.sleep(0.3)
        
        status_text.text("Scaling features...")
        progress.progress(50)
        time.sleep(0.3)
        
        status_text.text("Running prediction model...")
        progress.progress(75)
        time.sleep(0.3)
        
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        risk_score = None
        try:
            proba = model.predict_proba(scaled_input)[0][1]
            risk_score = round(float(proba) * 100, 1)
        except Exception:
            risk_score = None

        status_text.text("Calculating risk score...")
        progress.progress(100)
        time.sleep(0.2)
        
        status_text.empty()
        progress.empty()

    # Store in history
    st.session_state.prediction_history.append({
        'timestamp': datetime.now(),
        'risk_score': risk_score,
        'prediction': prediction,
        'age': age,
        'bp': resting_bp,
        'cholesterol': cholesterol
    })

    # ---- Risk category (badge) ----
    risk_label, risk_class, risk_emoji = get_risk_category(risk_score, prediction)
    badge_html = f"""
    <div style="margin-top:0.5rem; margin-bottom:0.8rem;">
        <span class="risk-badge {risk_class}">
            <span>{risk_emoji}</span>
            <span>{risk_label}</span>
        </span>
    </div>
    """

    with placeholder_result.container():
        st.markdown(badge_html, unsafe_allow_html=True)
        if prediction == 1:
            st.error("⚠️ **High Risk of Heart Disease Detected**")
            st.write(
                "Your inputs suggest a **higher likelihood of heart disease**. "
                "Please consult a **cardiologist or healthcare professional** for a detailed evaluation."
            )
            st.snow()
        else:
            st.success("✅ **Low Estimated Risk of Heart Disease**")
            st.write(
                "Based on the information provided, your **estimated risk appears low**. "
                "Still, regular check-ups and a heart-healthy lifestyle are very important."
            )
            st.balloons()

    # ---- Risk Gauge Chart ----
    with placeholder_gauge.container():
        gauge_value = risk_score if risk_score is not None else (80 if prediction == 1 else 10)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gauge_value,
            title={'text': "Estimated Risk (%)", 'font': {'size': 18, 'color': '#f1f5f9'}},
            number={'font': {'size': 36, 'color': '#f1f5f9'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#94a3b8'},
                'bar': {'thickness': 0.35, 'color': '#ec4899'},
                'bgcolor': 'rgba(15, 23, 42, 0.5)',
                'steps': [
                    {'range': [0, 20], 'color': "#065f46"},
                    {'range': [20, 50], 'color': "#713f12"},
                    {'range': [50, 100], 'color': "#7f1d1d"},
                ],
                'threshold': {
                    'line': {'color': "#f1f5f9", 'width': 3},
                    'thickness': 0.75,
                    'value': gauge_value
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#f1f5f9'},
            margin=dict(l=20, r=20, t=50, b=20),
            height=280,
            autosize=True  # Makes chart responsive
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Metrics + Health Score Card ----
    with placeholder_metrics.container():
        m1, m2, m3 = st.columns(3)
        if risk_score is not None:
            m1.metric("Estimated Risk", f"{risk_score}%")
        else:
            m1.metric("Estimated Risk", "N/A")
        m2.metric("Age", f"{age} yrs")
        m3.metric("Resting BP", f"{resting_bp} mm Hg")

    with placeholder_health.container():
        st.markdown("#### 🏥 Health Score Card")

        # Simple categorization (UI-only, not medical)
        def level_and_emoji(value, low_thr, high_thr):
            if value < low_thr:
                return "Good", "🟢"
            elif value < high_thr:
                return "Borderline", "🟡"
            else:
                return "High", "🔴"

        bp_level, bp_emoji = level_and_emoji(resting_bp, 120, 140)
        chol_level, chol_emoji = level_and_emoji(cholesterol, 200, 240)
        
        if max_hr >= 150:
            hr_level, hr_emoji = "Good", "🟢"
        elif max_hr >= 120:
            hr_level, hr_emoji = "Average", "🟡"
        else:
            hr_level, hr_emoji = "Low Capacity", "🔴"

        sugar_level = "Normal" if fasting_bs == 0 else "High"
        sugar_emoji = "🟢" if fasting_bs == 0 else "🔴"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"**Blood Pressure**\n\n{bp_emoji} {bp_level}\n\n`{resting_bp} mm Hg`")
        with c2:
            st.markdown(f"**Cholesterol**\n\n{chol_emoji} {chol_level}\n\n`{cholesterol} mg/dL`")
        with c3:
            st.markdown(f"**Max Heart Rate**\n\n{hr_emoji} {hr_level}\n\n`{max_hr} bpm`")
        with c4:
            st.markdown(f"**Blood Sugar**\n\n{sugar_emoji} {sugar_level}\n\n`FastingBS = {fasting_bs}`")

    # ---- Download Report (PDF or TXT) ----
    with placeholder_download.container():
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "=" * 50,
            "HEART STROKE RISK REPORT",
            "=" * 50,
            f"Generated: {report_time}",
            "",
            f"Risk Category: {risk_label}",
            f"Estimated Risk: {risk_score if risk_score is not None else 'N/A'}%",
            "",
            "INPUT DETAILS:",
            "-" * 50,
            f"Age: {age} years",
            f"Sex: {sex}",
            f"Resting Blood Pressure: {resting_bp} mm Hg",
            f"Cholesterol: {cholesterol} mg/dL",
            f"Fasting Blood Sugar: {'Yes (>120)' if fasting_bs == 1 else 'No (<120)'}",
            f"Max Heart Rate: {max_hr} bpm",
            f"Oldpeak: {oldpeak}",
            f"Chest Pain Type: {chest_pain}",
            f"Resting ECG: {resting_ecg}",
            f"Exercise-Induced Angina: {exercise_angina}",
            f"ST Slope: {st_slope}",
            "",
            "HEALTH INDICATORS:",
            "-" * 50,
            f"Blood Pressure: {bp_level} ({resting_bp} mm Hg)",
            f"Cholesterol: {chol_level} ({cholesterol} mg/dL)",
            f"Max Heart Rate: {hr_level} ({max_hr} bpm)",
            f"Blood Sugar: {sugar_level}",
            "",
            "DISCLAIMER:",
            "-" * 50,
            "This report is for educational purposes only and is NOT a",
            "medical diagnosis. Please consult with a qualified healthcare",
            "professional for proper medical advice and treatment.",
            "",
            "If you experience chest pain, shortness of breath, or other",
            "concerning symptoms, seek emergency medical care immediately.",
            "=" * 50
        ]
        report_text = "\n".join(lines)

        if PDF_AVAILABLE:
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 18)
                pdf.cell(0, 12, "Heart Stroke Risk Report", ln=True, align='C')
                pdf.ln(6)

                pdf.set_font("Arial", "", 10)
                pdf.set_text_color(80, 80, 80)
                pdf.cell(0, 6, f"Generated: {report_time}", ln=True, align='C')
                pdf.ln(8)

                pdf.set_font("Arial", "B", 14)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(0, 8, "Risk Assessment", ln=True)
                pdf.set_font("Arial", "", 11)
                pdf.multi_cell(0, 6, f"Risk Category: {risk_label}")
                pdf.multi_cell(0, 6, f"Estimated Risk Score: {risk_score if risk_score is not None else 'N/A'}%")
                pdf.ln(4)

                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 8, "Input Details", ln=True)
                pdf.set_font("Arial", "", 10)
                
                input_details = [
                    f"Age: {age} years",
                    f"Sex: {sex}",
                    f"Resting Blood Pressure: {resting_bp} mm Hg",
                    f"Cholesterol: {cholesterol} mg/dL",
                    f"Fasting Blood Sugar: {'Yes (>120)' if fasting_bs == 1 else 'No (<120)'}",
                    f"Max Heart Rate: {max_hr} bpm",
                    f"Oldpeak: {oldpeak}",
                    f"Chest Pain Type: {chest_pain}",
                    f"Resting ECG: {resting_ecg}",
                    f"Exercise-Induced Angina: {exercise_angina}",
                    f"ST Slope: {st_slope}"
                ]
                
                for detail in input_details:
                    pdf.multi_cell(0, 5, detail)
                pdf.ln(4)

                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 8, "Health Indicators", ln=True)
                pdf.set_font("Arial", "", 10)
                
                health_indicators = [
                    f"Blood Pressure: {bp_level} ({resting_bp} mm Hg)",
                    f"Cholesterol: {chol_level} ({cholesterol} mg/dL)",
                    f"Max Heart Rate: {hr_level} ({max_hr} bpm)",
                    f"Blood Sugar: {sugar_level}"
                ]
                
                for indicator in health_indicators:
                    pdf.multi_cell(0, 5, indicator)
                pdf.ln(6)

                pdf.set_font("Arial", "B", 12)
                pdf.set_text_color(200, 0, 0)
                pdf.cell(0, 8, "IMPORTANT DISCLAIMER", ln=True)
                pdf.set_font("Arial", "", 9)
                pdf.set_text_color(80, 80, 80)
                
                disclaimer_text = (
                    "This report is for educational purposes only and is NOT a medical diagnosis. "
                    "Please consult with a qualified healthcare professional for proper medical "
                    "advice and treatment. If you experience chest pain, shortness of breath, or "
                    "other concerning symptoms, seek emergency medical care immediately."
                )
                pdf.multi_cell(0, 5, disclaimer_text)
                
                pdf_bytes = pdf.output(dest="S").encode("latin-1")

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📄 Download Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"heart_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                with col2:
                    st.download_button(
                        "📝 Download Report (TXT)",
                        data=report_text.encode("utf-8"),
                        file_name=f"heart_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            except Exception as e:
                st.warning(f"PDF generation failed: {str(e)}. Offering text format only.")
                st.download_button(
                    "📝 Download Report (TXT)",
                    data=report_text.encode("utf-8"),
                    file_name=f"heart_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        else:
            st.download_button(
                "📝 Download Report (TXT)",
                data=report_text.encode("utf-8"),
                file_name=f"heart_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

# ------------------------ PREDICTION HISTORY ------------------------ #
if len(st.session_state.prediction_history) > 1:
    st.markdown("---")
    st.markdown(
        """
        <div class="section-header" style="margin-top:1.5rem;">
            <span class="icon">📈</span>
            <span class="text-gradient-pink-blue">Your Risk Trend</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="gradient-line-pink-blue">', unsafe_allow_html=True)
    
    history_df = pd.DataFrame(st.session_state.prediction_history)
    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
    history_df['time_label'] = history_df['timestamp'].dt.strftime('%H:%M:%S')
    
    # Create trend chart
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=history_df['time_label'],
        y=history_df['risk_score'],
        mode='lines+markers',
        name='Risk Score',
        line=dict(color='#ec4899', width=3),
        marker=dict(size=10, color='#ec4899', line=dict(color='#fff', width=2))
    ))
    
    fig_trend.update_layout(
        title='Risk Score Over Time',
        xaxis_title='Time',
        yaxis_title='Risk Score (%)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.5)',
        font={'color': '#f1f5f9'},
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(range=[0, 100]),
        autosize=True  # Makes chart responsive
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Checks", len(st.session_state.prediction_history))
    with col2:
        avg_risk = history_df['risk_score'].mean()
        st.metric("Average Risk", f"{avg_risk:.1f}%" if pd.notna(avg_risk) else "N/A")
    with col3:
        latest_risk = history_df['risk_score'].iloc[-1]
        previous_risk = history_df['risk_score'].iloc[-2] if len(history_df) > 1 else latest_risk
        delta = latest_risk - previous_risk if pd.notna(latest_risk) and pd.notna(previous_risk) else 0
        st.metric("Latest vs Previous", f"{latest_risk:.1f}%" if pd.notna(latest_risk) else "N/A", f"{delta:+.1f}%")

# ------------------------ FOOTER ------------------------ #
st.markdown("---")

# Footer using columns for better layout
footer_col = st.columns([1, 3, 1])[1]

with footer_col:
    st.markdown(
        """
        <div style="text-align: center; margin-top: 2rem;">
            <h3 style="color: #e2e8f0; margin-bottom: 0.5rem;">Heart Stroke Risk Predictor</h3>
            <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6; margin-bottom: 2rem;">
                ⚕️ This tool is for <b>educational insight only</b> and is <b>not a diagnostic device</b>.<br>
                If you have chest pain, shortness of breath, or feel unwell, please seek emergency medical care immediately.
            </p>
            <p style="color: #e2e8f0; font-weight: 600; font-size: 1.05rem; margin-bottom: 1rem;">
                Connect With Me
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Social media buttons in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.link_button(
            "⭐ GitHub",
            "https://github.com/Asif-Siddique",
            use_container_width=True
        )
    
    with col2:
        st.link_button(
            "💼 LinkedIn",
            "https://www.linkedin.com/in/asif-siddique",
            use_container_width=True
        )
    
    with col3:
        st.link_button(
            "📧 Email Me",
            "mailto:contact.asif@gmail.com",
            use_container_width=True
        )
    
    st.markdown(
        """
        <div style="text-align: center; color: #64748b; font-size: 0.85rem; margin-top: 2rem; padding-bottom: 1rem;">
            Built with ❤️ using Streamlit & Machine Learning<br>
            © 2025 ASIF SIDDIQUE • All Rights Reserved
        </div>
        """,
        unsafe_allow_html=True
    )