"""Streamlit entrypoint for the locked six predictor deep SSI model."""

from pathlib import Path

import streamlit as st

from ssi_catboost_app_complete import SSIRiskModel


st.set_page_config(page_title="Deep SSI risk assessment", page_icon="🏥")
st.title("Deep SSI risk assessment")
st.caption("Six predictor CatBoost model · calibrated probability")

model_dir = Path(__file__).resolve().parent / "model_artifacts"
required = (
    model_dir / "final_catboost_6predictor.cbm",
    model_dir / "final_catboost_6predictor_config.json",
)
if not all(path.is_file() for path in required):
    st.error("Model files are missing. Add the locked .cbm and config JSON to model_artifacts before deployment.")
    st.stop()


@st.cache_resource
def load_model():
    return SSIRiskModel(model_dir)


try:
    model = load_model()
except Exception as exc:
    st.error(f"Model could not be loaded: {exc}")
    st.stop()

st.write("Enter all six measurements. Use the same units and definitions as the model development dataset.")
with st.form("patient"):
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
    l3 = st.number_input("L3", value=None, format="%.4f")
    plt2 = st.number_input("PLT2", value=None, format="%.4f")
    crp2 = st.number_input("CRP2", value=None, format="%.4f")
    delta_esr = st.number_input("ΔESR_2to3", value=None, format="%.4f")
    delta_neutrophils = st.number_input("ΔN%_2to3", value=None, format="%.4f")
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

st.caption("Applicable time window: preoperative through POD4–5. For clinical assessment support; not a standalone diagnosis.")
