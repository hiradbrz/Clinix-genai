import streamlit as st
import requests

# Streamlit UI for interacting with the summarization service
st.title("Clinix AI Summarization Service")

# Input fields for user
patient_id = st.text_input("Patient ID", placeholder="Enter patient ID")
record_text = st.text_area("Patient Record Text", placeholder="Enter the full text of the patient record here...")
model_type = st.selectbox("Select Model Type", ["openai", "huggingface"])

# Display additional field if using Hugging Face
hf_model_name = None
if model_type == "huggingface":
    hf_model_name = st.text_input("Hugging Face Model Name", value="facebook/bart-large-cnn")

if st.button("Summarize"):
    if not patient_id or not record_text:
        st.error("Please provide both the patient ID and record text.")
    else:
        payload = {
            "patient_id": patient_id,
            "record_text": record_text,
            "model_type": model_type,
            "hf_model_name": hf_model_name
        }
        
        # Send the request to the FastAPI service
        try:
            response = requests.post("http://127.0.0.1:8000/summarize", json=payload)
            if response.status_code == 200:
                summary = response.json().get("summary", "No summary available.")
                st.success("Summarization Completed!")
                st.write("**Summary:**")
                st.write(summary)
            else:
                st.error(f"An error occurred: {response.json().get('detail')}")
        except Exception as e:
            st.error(f"Failed to connect to the summarization service: {e}")
