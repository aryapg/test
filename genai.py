# import json
# import pandas as pd
# import matplotlib.pyplot as plt
# import openai

# with open('./stored_value.txt', 'r') as file:
#     stored_value = json.loads(file.read())

# # print(stored_value)
# openai.api_key = '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'

# data_dict = {
#    "company_id": [int(entry[0]) for entry in stored_value],
#     "company_name": [entry[1] for entry in stored_value],
#     "compliant_status": [entry[2] == "Compliant" for entry in stored_value]
# }
# df = pd.DataFrame(data_dict)

# df_text = df.to_string(index=False)

# questions = ["How many companies are present?", "How many companies are compliant?"]

# def generate_summary(question):
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": df_text + "\n\n" + question}
#         ]
#     )
#     return response['choices'][0]['message']['content']

# for question in questions:
#     summary = generate_summary(question)
#     print(f'Question: {question}\nSummary: {summary}\n')
# compliance_counts = df['compliant_status'].value_counts()
# plt.figure(figsize=(7, 7)) 
# plt.pie(compliance_counts, labels=['Compliant', 'Non-compliant'], autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'], startangle=140, textprops={'fontsize': 14})
# plt.title('Compliance Status of Companies', fontdict={"fontsize": 16}, pad=20)
# plt.show()

import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import openai

openai.api_key = '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'
# Load stored data
with open('./stored_value.txt', 'r') as file:
    stored_value = json.loads(file.read())

# Convert stored data into DataFrame
data_dict = {
    "company_id": [int(entry[0]) for entry in stored_value],
    "company_name": [entry[1] for entry in stored_value],
    "compliant_status": [entry[2] == "Compliant" for entry in stored_value]
}
df = pd.DataFrame(data_dict)

# Function to generate summary using OpenAI
def generate_summary(question):
    df_text = df.to_string(index=False)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": df_text + "\n\n" + question}
        ]
    )
    return response['choices'][0]['message']['content']

# Streamlit App
st.title("Company Data Analysis")

# Display the DataFrame
st.write("## Company Data:")
st.write(df)

# Ask questions
question = st.text_input("Ask a question:")
if question:
    summary = generate_summary(question)
    st.write(f"### Question: {question}")
    st.write(f"*Summary:* {summary}")

# Button to show Compliance Status plot
if st.button("Show Compliance Status Plot"):
    st.write("## Compliance Status of Companies:")
    compliance_counts = df['compliant_status'].value_counts()

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot the pie chart on the axes
    ax.pie(compliance_counts, labels=['Compliant', 'Non-compliant'], autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'], startangle=140, textprops={'fontsize': 14})
    ax.set_title('Compliance Status of Companies', fontdict={"fontsize": 16}, pad=20)

    # Display the plot using st.pyplot()
    st.pyplot(fig)
