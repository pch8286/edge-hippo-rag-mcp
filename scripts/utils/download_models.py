import os
from sentence_transformers import SentenceTransformer

def download_model():
    model_name = "intfloat/multilingual-e5-small"
    print(f"Downloading {model_name}...")
    # This will cache the model in ~/.cache/huggingface/hub or local
    model = SentenceTransformer(model_name)
    print(f"Model {model_name} downloaded successfully.")
    
    # Save to a specific directory if needed for offline usage, 
    # but strictly speaking SentenceTransformers handles caching well.
    # For edge deployment, we might want to export to ONNX here later.

if __name__ == "__main__":
    download_model()
