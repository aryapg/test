import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap

# Title and description
st.title('🛡️ Company Compliance Checker 🕵️')
st.write('Hello! Please upload a dataset to check company compliance.')

# File upload
uploaded_file = st.file_uploader('Upload Dataset', type=['xlsx', 'xls'])

if uploaded_file:
    st.write('Selected File:', uploaded_file.name)
    df = pd.read_excel(uploaded_file)
    
    # Preprocess data
    label_encoder = LabelEncoder()
    categorical_columns = ['TransactionType', 'IncidentSeverity']
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])

    # Load pre-trained model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Assuming X and y are pre-defined from your training data
    X = df.drop('ComplianceStatus', axis=1)
    y = df['ComplianceStatus']

    # Train the model on the entire dataset
    rf_model.fit(X, y)

    # Make predictions on the same dataset
    y_pred = rf_model.predict(X)

    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    st.write(f'Accuracy: {accuracy:.4f}')

    # Display classification report
    st.write('Classification Report:')
    st.write(classification_report(y, y_pred))

    # Explain model predictions
    explainer = shap.Explainer(rf_model)
    sample_pred_proba = rf_model.predict_proba(X)
    sample_pred = rf_model.predict(X)
    shap_values = explainer.shap_values(X.iloc[0])
    feature_importance = np.abs(shap_values).mean(axis=0)
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    feature_importance_df_sorted = feature_importance_df.sort_values(by='Importance', ascending=False)
    top_features = feature_importance_df_sorted['Feature'].head(3).tolist()

    # Display top features and summary text
    st.write('Top Features:')
    for i, feature in enumerate(top_features):
        st.write(f'{i+1}. {feature}')
    
    summary_text = "Based on our analysis, the major factors affecting compliance status are:\n"
    for i, feature in enumerate(top_features):
        if feature_importance_df_sorted.loc[feature_importance_df_sorted['Feature'] == feature, 'Importance'].values[0] > 0:
            effect = "Improving"
        else:
            effect = "decreasing"
        summary_text += f"{i+1}. {feature}: {effect} this feature would contribute to compliance.\n"

    st.write('Summary Text:')
    st.write(summary_text)
