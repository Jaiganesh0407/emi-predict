# app.py
import streamlit as st
import pandas as pd
import numpy as np
import gzip, pickle, os
from pathlib import Path

@st.cache_resource
def load_artifacts(models_dir="models"):
    def gz_load(p):
        with gzip.open(p, 'rb') as f:
            return pickle.load(f)
    meta = gz_load(os.path.join(models_dir, 'meta_encoders.pkl.gz'))
    scaler = gz_load(os.path.join(models_dir, 'scaler.pkl.gz'))
    clf = gz_load(os.path.join(models_dir, 'classification_model.pkl.gz'))
    reg = gz_load(os.path.join(models_dir, 'regression_model.pkl.gz'))
    return meta, scaler, clf, reg

st.set_page_config(page_title="EMIPredict AI", layout="wide")
st.title("EMIPredict AI — EMI Eligibility & Max EMI Predictor")
st.markdown("Lightweight demo — model files expected in `models/`")

# Load
models_dir = st.text_input("Models directory", value="models")
try:
    meta, scaler, clf, reg = load_artifacts(models_dir=models_dir)
except Exception as e:
    st.error("Could not load models from '{}'. Run training script first. Error: {}".format(models_dir, e))
    st.stop()

label_encoders = meta['label_encoders']
target_le = meta['target_le']

# Input form
with st.form("input_form"):
    st.subheader("Applicant Financial Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=30)
        gender = st.selectbox("Gender", ["Male","Female"])
        marital_status = st.selectbox("Marital Status", ["Single","Married"])
        education = st.selectbox("Education", ["High School","Graduate","Post Graduate","Professional"])
        employment_type = st.selectbox("Employment Type", ["Private","Government","Self-employed"])
        years_of_employment = st.number_input("Years of Employment", min_value=0, max_value=50, value=3)
    with col2:
        monthly_salary = st.number_input("Monthly Salary (INR)", min_value=5000, max_value=1000000, value=50000)
        current_emi_amount = st.number_input("Existing EMI Amount", value=0)
        credit_score = st.slider("Credit Score", 300, 850, 700)
        bank_balance = st.number_input("Bank Balance", value=20000)
        emergency_fund = st.number_input("Emergency Fund", value=10000)
        existing_loans = st.selectbox("Existing Loans", ["Yes","No"])
    with col3:
        monthly_rent = st.number_input("Monthly Rent", value=5000)
        family_size = st.number_input("Family Size", min_value=1, max_value=20, value=3)
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)
        requested_amount = st.number_input("Requested Loan Amount", value=50000)
        requested_tenure = st.number_input("Requested Tenure (months)", value=12)
        emi_scenario = st.selectbox("EMI Scenario", ["E-commerce","Home Appliances","Vehicle","Personal Loan","Education"])
    submitted = st.form_submit_button("Predict")

if submitted:
    # build dataframe same as training features
    input_dict = {
        'gender': gender, 'marital_status': marital_status, 'education': education, 'employment_type': employment_type,
        'company_type': 'Unknown', 'house_type': 'Own', 'existing_loans': existing_loans, 'emi_scenario': emi_scenario,
        'age': age, 'monthly_salary': monthly_salary, 'years_of_employment': years_of_employment, 'monthly_rent': monthly_rent,
        'family_size': family_size, 'dependents': dependents, 'school_fees': 0.0, 'college_fees': 0.0, 'travel_expenses': 0.0,
        'groceries_utilities': 0.0, 'other_monthly_expenses': 0.0, 'current_emi_amount': current_emi_amount, 'credit_score': credit_score,
        'bank_balance': bank_balance, 'emergency_fund': emergency_fund, 'requested_amount': requested_amount, 'requested_tenure': requested_tenure
    }
    df = pd.DataFrame([input_dict])
    # create derived features same as preprocessing
    df['total_expenses'] = (df['monthly_rent'] + df['school_fees'] + df['college_fees'] + df['travel_expenses'] +
                            df['groceries_utilities'] + df['other_monthly_expenses']).fillna(0)
    df['debt_to_income_ratio'] = (df['current_emi_amount'] / (df['monthly_salary'] + 1e-6)) * 100
    df['expense_to_income_ratio'] = (df['total_expenses'] / (df['monthly_salary'] + 1e-6)) * 100
    df['available_income'] = df['monthly_salary'] - df['total_expenses'] - df['current_emi_amount']
    df['affordability_ratio'] = (df['available_income'] / (df['monthly_salary'] + 1e-6)) * 100
    df['emergency_fund_ratio'] = df['emergency_fund'] / (df['monthly_salary'] + 1e-6)
    df['dependents_income_ratio'] = df['dependents'] / (((df['monthly_salary']) / 10000) + 1e-6)

    # apply label encoders
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))

    # select features and scale numeric
    categorical = ['gender','marital_status','education','employment_type','company_type','house_type','existing_loans','emi_scenario']
    numeric = ['age','monthly_salary','years_of_employment','monthly_rent','family_size','dependents',
               'school_fees','college_fees','travel_expenses','groceries_utilities','other_monthly_expenses',
               'current_emi_amount','credit_score','bank_balance','emergency_fund','requested_amount','requested_tenure',
               'total_expenses','debt_to_income_ratio','expense_to_income_ratio','available_income','affordability_ratio',
               'emergency_fund_ratio','dependents_income_ratio']
    X = df[categorical + numeric].copy()
    X[numeric] = scaler.transform(X[numeric])

    # predictions
    clf_pred = clf.predict(X)[0]
    clf_proba = clf.predict_proba(X)[0] if hasattr(clf, "predict_proba") else None
    clf_label = target_le.inverse_transform([int(clf_pred)])[0]

    reg_pred = reg.predict(X)[0]

    st.success("Predictions ready!")
    st.subheader("Eligibility")
    st.write(f"Predicted class: **{clf_label}**")
    if clf_proba is not None:
        probs = {target_le.inverse_transform([i])[0]: float(p) for i,p in enumerate(clf_proba)}
        st.write("Class probabilities:")
        st.json(probs)
    st.subheader("Max Monthly EMI (INR)")
    st.write(f"Estimated safe max EMI: **₹ {reg_pred:,.0f}**")

    st.markdown("---")
    st.subheader("Model Summary")
    try:
        import json
        with open(os.path.join(models_dir,'training_meta.json')) as f:
            meta = json.load(f)
        st.json(meta)
    except Exception:
        pass
