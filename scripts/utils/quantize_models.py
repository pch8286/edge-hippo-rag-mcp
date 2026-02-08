#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quantizer")

def quantize_model(model_id: str, output_path: Path):
    logger.info(f"Quantizing {model_id} to {output_path}...")
    
    # 1. Export to ONNX (if not already)
    # ORTModelForFeatureExtraction handles export + loading
    logger.info("Loading and Exporting to ONNX...")
    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 2. Saving ONNX model
    save_dir = output_path / model_id.split("/")[-1]
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"Saved ONNX model to {save_dir}")
    
    # 3. Dynamic Quantization (Robust Quality + Pure ONNX Inference path)
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from optimum.onnxruntime import ORTQuantizer
    
    logger.info("Applying Dynamic Quantization (INT8)...")
    # Dynamic Quantization does not require calibration data
    
    # Configure Quantization: Dynamic INT8
    # is_static=False
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    
    quantizer = ORTQuantizer.from_pretrained(save_dir)
    
    quantized_dir = save_dir / "quantized"
    quantizer.quantize(
        save_dir=quantized_dir,
        quantization_config=qconfig
    )
    tokenizer.save_pretrained(quantized_dir)
    
    logger.info(f"Dynamic Quantization complete! Saved to {quantized_dir}")

def main():
    parser = argparse.ArgumentParser(description="Model Quantization Tool (ONNX INT8)")
    parser.add_argument("--model", type=str, default="intfloat/multilingual-e5-small", help="HuggingFace model ID")
    parser.add_argument("--output", type=str, default="models_quantized", help="Output directory")
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)
    
    quantize_model(args.model, output_path)

if __name__ == "__main__":
    main()
