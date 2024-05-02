import streamlit as st
from io import BytesIO
from PIL import Image
import sqlite3
import pandas as pd

# Function to check compliance using SQL
def check_compliance_sql(file):
    conn = sqlite3.connect('compliance_status.db')  # Connect to SQLite database
    df = pd.read_excel(file)  # Assuming the uploaded file is an Excel file
    df.to_sql('compliance_data', conn, if_exists='replace', index=False)  # Load data into SQLite table

    query = """
    SELECT * FROM compliance_data
    """  # Example SQL query, you can modify this as needed

    result = pd.read_sql(query, conn)  # Execute SQL query
    conn.close()  # Close database connection
    return result

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
                compliance_result = check_compliance_sql(selected_file)

            if compliance_result is not None:
                st.success("File uploaded successfully!")
                st.dataframe(compliance_result)  # Display SQL query result as DataFrame
            else:
                st.error("An error occurred while checking compliance.")
    
if __name__ == "__main__":
    main()
