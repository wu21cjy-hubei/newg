"""Streamlit entrypoint for the locked six predictor deep SSI model."""

from pathlib import Path

import streamlit as st

from ssi_catboost_app_complete import SSIRiskModel


st.set_page_config(page_title="Deep SSI risk assessment", page_icon="🏥")
st.title("Deep SSI risk assessment")

model_dir = Path(__file__).resolve().parent
required = (
    model_dir / "final_catboost_6predictor.cbm",
    model_dir / "final_catboost_6predictor_config.json",
)
if not all(path.is_file() for path in required):
    st.error("Model files are missing. Add the locked .cbm and config JSON to the repository root before deployment.")
    st.stop()


@st.cache_resource
def load_model():
    return SSIRiskModel(model_dir)


try:
    model = load_model()
except Exception as exc:
    st.error(f"Model could not be loaded: {exc}")
    st.stop()

st.write("Please enter the following six items of patient information")
with st.form("patient"):
    sex = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    l3 = st.number_input("Lymphocyte count (postoperative days 4–5, ×10⁹/L)", value=None, format="%.4f")
    plt2 = st.number_input("Platelet count (postoperative days 1–2, ×10⁹/L)", value=None, format="%.4f")
    crp2 = st.number_input("C-reactive protein (CRP, postoperative days 1–2, mg/L)", value=None, format="%.4f")
    delta_esr = st.number_input("ESR change value: ESR on days 7–8 – ESR on days 4–5 (mm/h)", value=None, format="%.4f")
    delta_neutrophils = st.number_input("Change in neutrophil percentage: N% (Days 7–8) − N% (Days 4–5) (percentage points)", value=None, format="%.4f")
    submitted = st.form_submit_button("Calculate risk", type="primary")

if submitted:
    values = [l3, plt2, crp2, delta_esr, delta_neutrophils]
    if any(value is None for value in values):
        st.error("Please enter every predictor.")
    else:
        patient = {
            "sex": sex,
            "L3": l3,
            "PLT2": plt2,
            "CRP2": crp2,
            "ΔESR_2to3": delta_esr,
            "ΔN%_2to3": delta_neutrophils,
        }
        try:
            result = model.predict(patient)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            st.subheader("Result")
            st.write(f"**Risk level:** {result.risk_level}")
            st.write(f"**Estimated probability of deep SSI:** {result.estimated_probability * 100:.1f}%")
            st.info(result.message_en)

st.caption("For clinical assessment support; not a standalone diagnosis.")
