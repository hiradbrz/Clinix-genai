#!/bin/bash

# Start the FastAPI service in the background
uvicorn api.summarization_service:app --host 0.0.0.0 --port 8000 &

# Start the Streamlit UI
streamlit run ui/summarization_ui.py --server.port 8501 --server.address 0.0.0.0
