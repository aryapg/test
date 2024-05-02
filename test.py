import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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
    'KYC': np.random.choice([0, 1], size=num_rows),
    'Privacy_Policy_Compliance': np.random.choice([0, 1], size=num_rows),
    'Data_Encryption': np.random.choice([0, 1], size=num_rows),
    'Capital_Adequacy_Ratio_CAR': np.random.uniform(8, 15, size=num_rows),
    'Basel_III_Compliance': np.random.choice([0, 1], size=num_rows),
    'Reporting_to_Regulatory_Authorities': np.random.choice([0, 1], size=num_rows),
    'Liquidity_Risk_Management': np.random.choice([0, 1], size=num_rows),
    'Interest_Rate_Risk_Management': np.random.choice([0, 1], size=num_rows),
    'Fair_Lending_Practices': np.random.choice([0, 1], size=num_rows),
    'Risk_Based_Internal_Audits': np.random.choice([0, 1], size=num_rows),
    'Cybersecurity_Policies_and_Procedures': np.random.choice([0, 1], size=num_rows),
    'Insider_Trading_Prevention': np.random.choice([0, 1], size=num_rows),
    'Compliant_Sales_Practices': np.random.choice([0, 1], size=num_rows),
    'clear_accurate_info': np.random.choice([0, 1], size=num_rows),
    'effective_complaint_handling': np.random.choice([0, 1], size=num_rows),
    'suitability_financial_products': np.random.choice([0, 1], size=num_rows),
    'data_privacy_security': np.random.choice([0, 1], size=num_rows),
    'product_approval_adherence': np.random.choice([0, 1], size=num_rows),
    'customer_satisfaction_index': np.random.uniform(70, 100, size=num_rows),
    'complaint_resolution_time_days': np.random.uniform(1, 30, size=num_rows),
    'product_suitability_score': np.random.uniform(0, 10, size=num_rows),
    'data_security_expenditure_perc': np.random.uniform(5, 15, size=num_rows),
    'product_approval_rate_perc': np.random.uniform(80, 100, size=num_rows),
    'marketing_compliance': np.random.choice([0, 1], size=num_rows),
    'transparent_communication': np.random.choice([0, 1], size=num_rows),
    'advertisement_accuracy_score': np.random.uniform(0, 10, size=num_rows),
    'customer_communication_score': np.random.uniform(0, 10, size=num_rows),
    'stakeholder_engagement_score': np.random.uniform(0, 10, size=num_rows),
    'public_transparency_score': np.random.uniform(0, 10, size=num_rows),
    'social_media_compliance': np.random.choice([0, 1], size=num_rows),
    'regulatory_disclosure': np.random.choice([0, 1], size=num_rows),
    'TransactionAmount': np.random.uniform(1000, 100000, size=num_rows),
    'TransactionType': np.random.choice(['Deposit', 'Withdrawal'], size=num_rows),
    'IsSuspicious': np.random.choice([0, 1], size=num_rows),
    'EmployeeCount': np.random.randint(50, 500, size=num_rows),
    'CyberSecurityBudget': np.random.uniform(50000, 500000, size=num_rows),
    'IncidentSeverity': np.random.choice(['Low', 'Medium', 'High'], size=num_rows),
    'VulnerabilityCount': np.random.randint(0, 50, size=num_rows),
    'SolvencyRatio': np.random.uniform(0.1, 1.0, size=num_rows),
    'Audit_Committee_Existence': np.random.choice([0, 1], size=num_rows),
    'Internal_Audit_Function': np.random.choice([0, 1], size=num_rows),
    'Code_of_Ethics_Policy': np.random.choice([0, 1], size=num_rows),
    'Whistleblower_Policy': np.random.choice([0, 1], size=num_rows),
    'Risk_Management_Framework': np.random.choice([0, 1], size=num_rows),
    'Conflict_of_Interest_Disclosure': np.random.choice([0, 1], size=num_rows),
    'Related_Party_Transactions_Monitoring': np.random.choice([0, 1], size=num_rows),
    'Executive_Compensation_Disclosure': np.random.choice([0, 1], size=num_rows),
    'Shareholder_Rights_Protection': np.random.choice([0, 1], size=num_rows),
    'Governance_Policies_Disclosure': np.random.choice([0, 1], size=num_rows),
    'Succession_Planning': np.random.choice([0, 1], size=num_rows),
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
