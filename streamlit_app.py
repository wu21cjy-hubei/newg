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

st.write("请输入以下六项患者信息。")
with st.form("patient"):
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    l3 = st.number_input("淋巴细胞计数（术后第4–5天，×10⁹/L）", value=None, format="%.4f")
    plt2 = st.number_input("血小板计数（术后第1–2天，×10⁹/L）", value=None, format="%.4f")
    crp2 = st.number_input("C反应蛋白（CRP，术后第1–2天，mg/L）", value=None, format="%.4f")
    delta_esr = st.number_input("血沉变化值：第7–8天 ESR − 第4–5天 ESR（mm/h）", value=None, format="%.4f")
    delta_neutrophils = st.number_input("中性粒细胞百分比变化值：第7–8天 N% − 第4–5天 N%（百分点）", value=None, format="%.4f")
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
