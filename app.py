import streamlit as st
import pandas as pd
import numpy as np
from feature_engine import discretisation, encoding

model = pd.read_pickle("model_heart_dis.pkl")

st.title("❤️ Heart Disease Risk Calculator")
st.markdown("### A Data Science Portfolio Project")

st.warning(
    "⚠️ **DISCLAIMER:** This application is for **educational purposes only**. It does not replace a professional medical diagnosis. If you have health concerns, please consult a doctor."
)

st.divider()
st.header("Patient Information")

expander = st.expander("**Variable Descriptions (Click to learn more)**")
with expander:
    st.markdown(
        """
        **1. Age:**
        The patient's age in years. Risk of heart disease generally increases with age.

        **2. Biological Sex:**
        Biological sex assigned at birth (Male or Female). Statistics show different risk profiles for men and women.
        <br>
        <br>
        **A Quick Note on Gender and Data** 
        
        We want everyone to feel welcome and included when using this tool. 
        <br>
        This risk model requires you to select **Male** or **Female** because the **original medical research and datasets** used for 
        training are limited to these two biological categories.
        <br>
        This input is used **only** for calculating established biological risk factors (like hormonal influences and risk onset age), and 
        **in no way** is intended to exclude or ignore the diversity of gender identities. 
        <br>
        We appreciate your understanding of this limitation in the source data and are committed to inclusivity.

        **3. Chest Pain Type:**
        * **Typical Angina:** Chest pain caused by reduced blood flow to the heart (usually feels like pressure/squeezing).
        * **Atypical Angina:** Chest pain that doesn't fit the "classic" description but is still suspicious.
        * **Non-anginal Pain:** Pain not related to the heart (e.g., muscle strain, rib pain).
        * **Asymptomatic:** No chest pain present.

        **4. Resting Blood Pressure:**
        The top number (systolic) of your blood pressure reading when sitting quietly (mm Hg).
        * *Normal:* < 120
        * *Elevated:* 120-129
        * *Hypertension:* > 130

        **5. Serum Cholesterol:**
        Total cholesterol level in the blood (mg/dL). High levels can lead to plaque buildup in arteries.
        * *Desirable:* < 200
        * *High:* > 240

        **6. Max Heart Rate Achieved:**
        The highest number of heartbeats per minute reached during maximum physical exertion (like running on a treadmill).

        **7. Exercise Induced Angina:**
        Do you feel chest pain or tightness specifically when you exercise or exert yourself? (Yes/No).
        """,
        unsafe_allow_html=True,
    )

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("What is your age?", min_value=18, max_value=100, value=50)
    sex = st.radio("What is your biological sex?", ["Female", "Male"])
    sex_bin = 1 if sex == "Male" else 0

    chest_pain_opt = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
    )
    # Mapeamento reverso para o modelo
    cp_mapping = {
        "Typical Angina": 1,
        "Atypical Angina": 2,
        "Non-anginal Pain": 3,
        "Asymptomatic": 4,
    }
    chest_pain = cp_mapping[chest_pain_opt]

with col2:
    restbps = st.slider("Resting Blood Pressure (mm Hg)", 70, 200, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 70, 700, 200)
    max_heart_rate = st.slider("Max Heart Rate Achieved", 50, 220, 150)

    exc_angina = st.radio("Exercise Induced Angina?", ["No", "Yes"])
    exc_angina_bin = 1 if exc_angina == "Yes" else 0

# expander = st.expander("Advanced Medical Test Results (Optional/Estimated)")
# with expander:
#     oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 7.0, 0.0, step=0.1)
#
#     slope_opt = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
#     slope_mapping = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
#     slope = slope_mapping[slope_opt]
#
#     n_fl_maj_ves = st.slider("Major Vessels (Fluoroscopy)", 0, 3, 0)
#
#     thal_opt = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"])
#     thal_mapping = {"Normal": 3, "Fixed Defect": 6, "Reversable Defect": 7}
#     thal = thal_mapping[thal_opt]

user_input_dict = {
    "age": age,
    "sex": sex_bin,
    "chest_pain": chest_pain,
    "restbps": restbps,
    "chol": chol,
    "max_heart_rate": max_heart_rate,
    "exc_angina": exc_angina_bin,
    # 'oldpeak': oldpeak,
    # 'slope': slope,
    # 'n_fl_maj_ves': n_fl_maj_ves,
    # 'thal': thal
}

df = pd.DataFrame([user_input_dict])

proba = model["model"].predict_proba(df[model["features"]])[:, 1][0]
st.markdown("---")

if proba < 0.3:
    st.success(f"🟢 **Low Risk:** Likelihood of heart disease {proba:.1%}")
    st.write("**Great! Your profile suggests a healthy heart condition.**")

elif 0.3 <= proba < 0.7:
    st.warning(f"🟡 **Moderate Risk:** Likelihood of heart disease {proba:.1%}")
    st.write("**Attention:** Your profile shows some risk factors.")
    st.info(
        "**Recommendation**: Schedule a routine check-up with a cardiologist to be sure."
    )

else:
    st.error(f"🔴 **High Risk:** Likelihood of heart disease {proba:.1%}")
    st.write("**Alert:** Your profile strongly resembles patients with heart disease.")
    st.warning(
        "**Recommendation**: Please consult a doctor immediately for clinical exams."
    )
