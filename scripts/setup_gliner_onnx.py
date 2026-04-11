from seahorse.config import settings
from seahorse.model_setup import ensure_gliner_onnx_model

MODEL_ID = settings.GLINER_ONNX_REPO_ID
TARGET_DIR = settings.GLINER_ONNX_PATH or "models/gliner_onnx"

def download_model():
    print(f"Downloading {MODEL_ID} to {TARGET_DIR}...")
    try:
        ensure_gliner_onnx_model(TARGET_DIR, repo_id=MODEL_ID, force=False)
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")

if __name__ == "__main__":
    download_model()
