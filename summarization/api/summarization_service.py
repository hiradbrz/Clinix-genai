from typing import Optional
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import requests
import os

api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

# Initialize the FastAPI app
app = FastAPI()

# Your Hugging Face API URL and Headers
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": "Bearer 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"}

def query_huggingface(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(f"Hugging Face API error: {response.status_code}, {response.text}")

# Request format
class RecordInput(BaseModel):
    patient_id: str
    record_text: str
    model_type: str  # Options: "openai", "local", "huggingface"
    hf_model_name: Optional[str] = None  # Optional, required only if using a Hugging Face model

@app.post("/summarize")
async def summarize_record(record: RecordInput):
    try:
        if record.model_type == "openai":
            response = client.chat.completions.create(model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes medical records."},
                {"role": "user", "content": f"Summarize the following patient record: {record.record_text}"}
            ],
            max_tokens=150)
            summary = response.choices[0].message.content

        elif record.model_type == "huggingface":
            # Ensure hf_model_name is provided
            if not record.hf_model_name:
                raise ValueError("Hugging Face model name must be provided when using the 'huggingface' model type.")
            output = query_huggingface({
                "inputs": record.record_text
            })
            summary = output[0]['summary_text'] if output and 'summary_text' in output[0] else "No summary generated."

        else:
            raise ValueError("Invalid model_type specified. Choose 'openai', 'local', or 'huggingface'.")

        return {"patient_id": record.patient_id, "summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
