import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import shap

# Load the pre-trained model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Define LabelEncoder for categorical columns
label_encoder = LabelEncoder()

st.title('Company Compliance Checker')

uploaded_file = st.file_uploader("Upload Dataset", type=["xlsx", "xls"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write(df.head())

    X = df.drop(['ComplianceStatus', 'company_name'], axis=1)
    y = df['ComplianceStatus']
    y = label_encoder.fit_transform(y)

    categorical_columns = ['TransactionType', 'IncidentSeverity']
    for column in categorical_columns:
        X[column] = label_encoder.fit_transform(X[column])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'Accuracy: {accuracy:.4f}')

    st.write(classification_report(y_test, y_pred))

    explainer = shap.Explainer(rf_model)
    shap_values = explainer.shap_values(X.iloc[0])
    feature_importance = np.abs(shap_values).mean(axis=0)
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    feature_importance_df_sorted = feature_importance_df.sort_values(by='Importance', ascending=False)
    top_features = feature_importance_df_sorted['Feature'].head(3).tolist()
    summary_text = "Based on our analysis, the major factors affecting compliance status are:\n"
    for i, feature in enumerate(top_features):
        if feature_importance_df_sorted.loc[feature_importance_df_sorted['Feature'] == feature, 'Importance'].values[0] > 0:
            effect = "Improving"
        else:
            effect = "decreasing"
        summary_text += f"{i+1}. {feature}: {effect} this feature would contribute to compliance.\n"

    compliance_status = 'Compliant' if y_pred[0] >= 0.5 else 'Not Compliant'

    st.write({
        "complianceStatus": compliance_status,
        "topFeatures": top_features,
        "summaryText": summary_text
    })
