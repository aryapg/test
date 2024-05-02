import streamlit as st
import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pymysql
from faker import Faker

# Initialize Faker for generating fake data
fake = Faker()

# Number of rows in the synthetic dataset
num_rows = 10000

# Generate synthetic data for each feature
data = {
    'Customer_Due_Diligence_CDD': np.random.choice([0, 1], size=num_rows),
    'AML_Compliance': np.random.choice([0, 1], size=num_rows),
    # Add other synthetic data columns here
    'ComplianceStatus': np.random.choice(['Compliant', 'Non-Compliant'], size=num_rows),
}

# Create a DataFrame from the generated data
df = pd.DataFrame(data)

X = df.drop('ComplianceStatus', axis=1)
y = df['ComplianceStatus']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
st.write(f'Accuracy: {accuracy:.4f}')

st.write(classification_report(y_test, y_pred))

db = pymysql.connect(host='localhost',
                     user='aryapg',
                     password='Arya440022#@',
                     db='compliance_status',
                     charset='utf8mb4',
                     cursorclass=pymysql.cursors.DictCursor)

st.title('🛡️ Company Compliance Checker 🕵️')

uploaded_file = st.file_uploader("Upload a file", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        sample_data = df.drop(['ComplianceStatus', 'company_name'], axis=1)
        categorical_columns = ['TransactionType', 'IncidentSeverity']
        for column in categorical_columns:
            sample_data[column] = label_encoder.fit_transform(sample_data[column])
        sample_pred_proba = rf_model.predict(sample_data)
        sample_pred = (sample_pred_proba > 0.5).astype(int)
        custom_label_mapping = {0: 'Not compliant', 1: 'Compliant'}
        sample_pred_decoded = np.vectorize(custom_label_mapping.get)(sample_pred.flatten())
        st.write({"complianceStatus": sample_pred_decoded[0]})
        st.write({"topFeatures": 'Feature 1, Feature 2, Feature 3'})
        st.write({"summaryText": 'Summary text based on the analysis.'})
    except Exception as e:
        st.write({"error": str(e)})
