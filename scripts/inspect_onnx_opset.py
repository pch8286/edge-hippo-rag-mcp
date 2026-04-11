import onnx
import sys

model_path = "models/gliner_onnx/onnx/encoder_fp16.onnx"

try:
    print(f"Loading {model_path}...")
    model = onnx.load(model_path)
    print("Model loaded.")
    
    print(f"Opset Import:")
    for opset in model.opset_import:
        print(f" - Domain: {opset.domain}, Version: {opset.version}")
        
    print("Checking model...")
    onnx.checker.check_model(model)
    print("Model checked successfully.")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
