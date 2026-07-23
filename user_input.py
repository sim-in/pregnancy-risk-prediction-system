import pandas as pd
import joblib


def get_float(question):
    while True:
        try:
            return float(input(question))
        except ValueError:
            print("Please enter a number.")


def get_int(question):
    while True:
        try:
            return int(input(question))
        except ValueError:
            print("Please enter a whole number.")


def yes_no(question):
    while True:
        answer = input(question + " (yes/no): ").strip().lower()

        if answer == "yes":
            return 1
        elif answer == "no":
            return 0
        else:
            print("Please type only yes or no.")


def calculate_bmi(height_feet, height_inch, weight_kg):
    total_inches = (height_feet * 12) + height_inch
    height_meter = total_inches * 0.0254

    bmi = weight_kg / (height_meter ** 2)
    return round(bmi, 2)


def calculate_risk_percentage(model, user_data):
    probabilities = model.predict_proba(user_data)[0]
    classes = model.named_steps["model"].classes_

    risk_probability = probabilities[list(classes).index(1)]

    risk_percentage = risk_probability * 100

    if risk_percentage < 0:
        risk_percentage = 0

    if risk_percentage > 100:
        risk_percentage = 100

    return round(risk_percentage, 2)


model_1 = joblib.load("best_model_dataset_1.pkl")


print("Pregnancy Risk Prediction - Dataset 1 Model")
print("Please enter the information below.\n")


age = get_int("Age: ")

height_feet = get_int("Height - feet: ")
height_inch = get_float("Height - inch: ")
weight_kg = get_float("Weight in kg: ")

bmi = calculate_bmi(height_feet, height_inch, weight_kg)

systolic_bp = get_float("Systolic blood pressure: ")
diastolic_bp = get_float("Diastolic blood pressure: ")

blood_sugar = get_float("Blood sugar level in mmol/L, for example 6.5: ")

body_temp = get_float("Body temperature in Fahrenheit: ")
heart_rate = get_float("Heart rate: ")

previous_complications = yes_no("Any previous pregnancy complications?")
pre_existing_diabetes = yes_no("Pre-existing diabetes?")
gestational_diabetes = yes_no("Gestational diabetes?")
mental_health = yes_no("Any mental health stress, anxiety, or depression?")


user_data_1 = pd.DataFrame([{
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


risk_1 = calculate_risk_percentage(model_1, user_data_1)


print("\n==============================")
print("Calculated BMI:", bmi)
print("Dataset 1 pregnancy risk:", risk_1, "%")
print("==============================")

print("\nNote: This is only a machine learning prediction, not a medical diagnosis.")