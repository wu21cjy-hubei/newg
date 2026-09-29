"""English POD7–8 risk assessment app."""
import streamlit as st
from pod78_model import POD78Model

st.set_page_config(page_title="Deep SSI risk assessment — POD7–8", page_icon="🏥")
st.title("Deep SSI risk assessment")
st.write("Please enter the following variables available up to postoperative days 7–8.")


@st.cache_resource
def load_model():
    return POD78Model()


try:
    model = load_model()
except Exception:
    st.error("The prediction model could not be loaded. Please contact the app administrator.")
    st.stop()

with st.form("pod78_patient"):
    sex = st.selectbox("Gender", [0, 1], format_func=lambda v: "Female" if v == 0 else "Male")
    lymphocytes = st.number_input("Lymphocyte count (postoperative days 4–5, ×10⁹/L)", min_value=0.0, value=None, format="%.4f")
    neutrophils = st.number_input("Neutrophil percentage (postoperative days 7–8, %)", min_value=0.0, max_value=100.0, value=None, format="%.4f")
    platelets = st.number_input("Platelet count (postoperative days 1–2, ×10⁹/L)", min_value=0.0, value=None, format="%.4f")
    platelet_change = st.number_input("Platelet count change: postoperative days 1–2 – preoperative value (×10⁹/L)", value=None, format="%.4f")
    crp_change = st.number_input("CRP change: postoperative days 7–8 – postoperative days 1–2 (mg/L)", value=None, format="%.4f")
    monocyte_change = st.number_input("Monocyte percentage change: postoperative days 7–8 – postoperative days 4–5 (percentage points)", min_value=-100.0, max_value=100.0, value=None, format="%.4f")
    submitted = st.form_submit_button("Calculate risk", type="primary")

if submitted:
    values = [lymphocytes, neutrophils, platelets, platelet_change, crp_change, monocyte_change]
    if any(value is None for value in values):
        st.error("Please enter every variable.")
    else:
        patient = dict(zip(model.config["feature_columns"], [sex] + values))
        try:
            result = model.predict(patient)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.subheader("Result")
            st.write(f"**Risk level:** {result['risk_level']}")
            st.write(f"**Estimated probability of deep SSI:** {result['estimated_probability'] * 100:.1f}%")
            st.info(result["message"])

st.caption("For clinical assessment support; not a standalone diagnosis.")
