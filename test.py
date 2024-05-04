import streamlit as st
import requests
import pandas as pd
from io import BytesIO

# Title and description
st.title("Company Compliance Checker")
st.write("Hello! Upload a dataset to check company compliance.")

# File upload
uploaded_file = st.file_uploader("Upload Dataset", type=['xlsx', 'xls'])
if uploaded_file is not None:
    st.write('File uploaded successfully!')
    df = pd.read_excel(uploaded_file)

    # Display uploaded file data
    st.write(df)

    # Check compliance button
    if st.button('Check Compliance'):
        with st.spinner('Checking compliance...'):
            # Prepare the file for upload
            files = {'file': ('dataset.xlsx', BytesIO(uploaded_file.read()), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}

            # Send request to Flask backend
            response = requests.post('http://localhost:5000/', files=files)

            # Handle response from backend
            if response.status_code == 200:
                result = response.json()
                st.success(result['complianceStatus'])
                st.write('Top Features:')
                st.write(result['topFeatures'])
                st.write('Summary Text:')
                st.write(result['summaryText'])
            else:
                st.error('Error checking compliance. Please try again.')

if __name__ == '__main__':
    app.run(debug=True,host='localhost', port=5000, debug=True)
