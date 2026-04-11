import logging
import warnings
from pathlib import Path
from gliner import GLiNER

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quantize_gliner():
    model_id = "fastino/gliner2-multi-v1"
    output_dir = Path("models_quantized/gliner_small")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading GLiNER model: {model_id}")
    try:
        model = GLiNER.from_pretrained(model_id)
        
        # Save Config and Tokenizer to destination first
        logger.info(f"Saving config and tokenizer to {output_dir}...")
        model.save_pretrained(output_dir)
        
        # Remove PyTorch weights to save space (keep only ONNX)
        for f in output_dir.glob("*.bin"):
            f.unlink()
        for f in output_dir.glob("*.safetensors"):
            f.unlink()
            
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    logger.info(f"Exporting to ONNX (Quantized) at {output_dir}...")
    try:
        paths = model.export_to_onnx(
            save_dir=output_dir,
            onnx_filename="model.onnx",
            quantized_filename="model_quantized.onnx",
            quantize=True,
            opset=17 # Robust opset
        )
        logger.info(f"Export successful! Paths: {paths}")
        
    except NotImplementedError as e:
        logger.error(f"Model architecture not supported for built-in export: {e}")
        logger.info("Checking if custom export is needed...")
    except Exception as e:
        logger.error(f"Export failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quantize_gliner()
