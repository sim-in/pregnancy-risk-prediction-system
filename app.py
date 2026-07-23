import pandas as pd
import joblib
import streamlit as st


def calculate_bmi(height_feet, height_inch, weight_kg):
    total_inches = (height_feet * 12) + height_inch
    height_meter = total_inches * 0.0254

    if height_meter <= 0:
        return 0

    bmi = weight_kg / (height_meter ** 2)
    return round(bmi, 2)


def calculate_glucose_from_blood_sugar(blood_sugar_mmol):
    # Dataset 1 uses BS in mmol/L.
    # This glucose value is only shown to the user in mg/dL.
    glucose = blood_sugar_mmol * 18
    return round(glucose, 2)


def yes_no_to_number(answer):
    if answer == "Yes":
        return 1
    else:
        return 0


@st.cache_resource
def load_model():
    model = joblib.load("best_model_dataset_1.pkl")
    return model


def calculate_risk_percentage(model, user_data):
    probabilities = model.predict_proba(user_data)[0]
    classes = model.named_steps["model"].classes_

    risk_probability = 0

    for index in range(len(classes)):
        if classes[index] == 1:
            risk_probability = probabilities[index]

    risk_percentage = risk_probability * 100

    if risk_percentage < 0:
        risk_percentage = 0

    if risk_percentage > 100:
        risk_percentage = 100

    return round(risk_percentage, 2)


def risk_message(risk):
    if risk < 33:
        return "Low predicted risk"
    elif risk < 66:
        return "Medium predicted risk"
    else:
        return "High predicted risk"


st.set_page_config(
    page_title="Pregnancy Risk Prediction",
    layout="centered"
)

st.title("Pregnancy Risk Prediction")
st.write("This app uses the dataset_1 trained model to predict pregnancy risk percentage.")
st.markdown("[Source of dataset](https://www.kaggle.com/datasets/vmohammedraiyyan/maternal-health-and-high-risk-pregnancy-dataset)")
st.markdown("""
**Prepared by:**
* Ragib Shahriar Majid - 2311007
* Umme Habiba - 2311009
* Arafat Howlader - 2311026
""")

model = load_model()


with st.form("pregnancy_risk_form"):
    st.subheader("Personal Information")

    age = st.number_input("Age", min_value=10, max_value=70, value=25, step=1)

    col1, col2 = st.columns(2)
    with col1:
        height_feet = st.number_input("Height - feet", min_value=3, max_value=8, value=5, step=1)
    with col2:
        height_inch = st.number_input("Height - inch", min_value=0.0, max_value=11.9, value=4.0, step=0.1)

    weight_kg = st.number_input("Weight in kg", min_value=20.0, max_value=200.0, value=60.0, step=0.5)

    st.subheader("Medical Information")

    systolic_bp = st.number_input("Systolic blood pressure", min_value=60.0, max_value=250.0, value=120.0, step=1.0)
    diastolic_bp = st.number_input("Diastolic blood pressure", min_value=40.0, max_value=160.0, value=80.0, step=1.0)

    blood_sugar = st.number_input(
        "Blood sugar level in mmol/L, for example 6.5",
        min_value=1.0,
        max_value=30.0,
        value=6.5,
        step=0.1
    )

    body_temp = st.number_input(
        "Body temperature in Fahrenheit",
        min_value=90.0,
        max_value=110.0,
        value=98.0,
        step=0.1
    )

    heart_rate = st.number_input("Heart rate", min_value=30.0, max_value=220.0, value=80.0, step=1.0)

    st.subheader("Pregnancy and Health History")

    previous_complications_answer = st.selectbox("Any previous pregnancy complications?", ["No", "Yes"])
    pre_existing_diabetes_answer = st.selectbox("Pre-existing diabetes?", ["No", "Yes"])
    gestational_diabetes_answer = st.selectbox("Gestational diabetes?", ["No", "Yes"])
    mental_health_answer = st.selectbox("Any mental health stress, anxiety, or depression?", ["No", "Yes"])

    submit_button = st.form_submit_button("Predict Risk")


if submit_button:
    bmi = calculate_bmi(height_feet, height_inch, weight_kg)
    glucose = calculate_glucose_from_blood_sugar(blood_sugar)

    previous_complications = yes_no_to_number(previous_complications_answer)
    pre_existing_diabetes = yes_no_to_number(pre_existing_diabetes_answer)
    gestational_diabetes = yes_no_to_number(gestational_diabetes_answer)
    mental_health = yes_no_to_number(mental_health_answer)

    user_data = pd.DataFrame([{
    "Age": age,
    "Systolic BP": systolic_bp,
    "Diastolic": diastolic_bp,
    "BS": blood_sugar,
    "Body Temp": body_temp,
    "BMI": bmi,
    "Previous Complications": previous_complications,
    "Preexisting Diabetes": pre_existing_diabetes,
    "Gestational Diabetes": gestational_diabetes,
    "Mental Health": mental_health,
    "Heart Rate": heart_rate
}])

    risk = calculate_risk_percentage(model, user_data)

    st.divider()
    st.subheader("Prediction Result")

    st.metric("Pregnancy risk", str(risk) + " %")
    st.progress(float(risk/100))

    st.write("Risk level:", risk_message(risk))

    st.write("Calculated BMI:", bmi)
    st.write("Calculated glucose:", str(glucose) + " mg/dL")

    st.warning("This is only a machine learning prediction, not a medical diagnosis. Please consult a doctor for medical advice.")