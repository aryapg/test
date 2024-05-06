import streamlit as st
import requests
from io import BytesIO
from PIL import Image

# Function to check compliance
def check_compliance(file):
    url = 'https://compliance-backend.onrender.com'  # Update this URL with your server URL
    files = {'file': file}
    response = requests.post(url, files=files)
    return response.json()

# Streamlit app
def main():
    st.set_page_config(page_title="Company Compliance Checker")

    st.markdown("# 🛡️ Company Compliance Checker 🕵️")
    st.markdown("Hello! Please upload a dataset to check company compliance.")

    selected_file = st.file_uploader("Upload Dataset", type=['csv', 'xls', 'xlsx'])

    if selected_file:
        st.markdown(f"Selected File: {selected_file.name}")

        if st.button("Check Compliance"):
            with st.spinner("Checking Compliance..."):
                compliance_result = check_compliance(selected_file)

            if compliance_result:
                st.success("File uploaded successfully!")
                if 'complianceStatus' in compliance_result:
                    st.info(f"Compliance Status: {compliance_result['complianceStatus']}")
                if 'summaryText' in compliance_result and 'topFeatures' in compliance_result:
                    summary_lines = compliance_result['summaryText'].split('\n')
                    st.markdown("### Summary Text:")
                    for line in summary_lines:
                        st.write(line)
                    st.markdown("### Top Features:")
                    for feature in compliance_result['topFeatures']:
                        st.write(feature)
            else:
                st.error("An error occurred while checking compliance.")
    
if __name__ == "__main__":
    main()
