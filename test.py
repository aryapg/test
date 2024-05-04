import streamlit as st
import requests

# Function to check compliance
def check_compliance(file):
    url = 'http://192.168.0.1:5000/'  # Update with your server URL (use IP address)
    files = {'file': file}
    try:
        response = requests.post(url, files=files)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.ConnectionError as e:
        st.error(f"Error connecting to the server: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

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
                st.success("Compliance check completed successfully!")
                if 'complianceStatus' in compliance_result:
                    st.info(f"Compliance Status: {compliance_result['complianceStatus']}")
                if 'summaryText' in compliance_result and 'topFeatures' in compliance_result:
                    st.markdown("### Summary Text:")
                    st.write(compliance_result['summaryText'])
                    st.markdown("### Top Features:")
                    for feature in compliance_result['topFeatures']:
                        st.write(feature)
            else:
                st.error("An error occurred while checking compliance.")

if __name__ == "__main__":
    main()
