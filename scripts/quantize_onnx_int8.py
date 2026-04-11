#!/usr/bin/env python3
"""
Quantize GLiNER2 ONNX models to INT8 (dynamic quantization).

- span_rep.onnx and count_embed.onnx: FP32 originals exist → quantize directly
- encoder_fp16.onnx: FP16 only → convert to FP32 via onnxconverter-common, then quantize

Usage:
    .venv313/bin/python scripts/quantize_onnx_int8.py
"""
import os
import time
import shutil
from pathlib import Path

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


INPUT_DIR = Path("models/gliner_onnx/onnx")
OUTPUT_DIR = Path("models/gliner_onnx/onnx_int8")


def quantize_fp32_to_int8(fp32_path: str, int8_path: str):
    """Apply dynamic INT8 quantization to an FP32 model."""
    print(f"  Quantizing FP32 -> INT8: {os.path.basename(fp32_path)}")
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    
    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
    int8_size = os.path.getsize(int8_path) / (1024 * 1024)
    ratio = (1 - int8_size / fp32_size) * 100
    print(f"  Size: {fp32_size:.1f} MB -> {int8_size:.1f} MB ({ratio:.1f}% reduction)")


def convert_encoder_fp16_to_fp32(fp16_path: str, fp32_path: str):
    """Convert the encoder from FP16 to FP32. Handles all graph-level type refs."""
    from onnx import numpy_helper, TensorProto
    import numpy as np
    
    print(f"  Converting encoder FP16 -> FP32...")
    model = onnx.load(fp16_path)
    
    # 1. Convert all FP16 initializers to FP32
    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            new = numpy_helper.from_array(arr, name=init.name)
            init.CopyFrom(new)
    
    # 2. Fix all type annotations (inputs, outputs, value_info)
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        tt = vi.type.tensor_type
        if tt.elem_type == TensorProto.FLOAT16:
            tt.elem_type = TensorProto.FLOAT
    
    # 3. Fix Cast nodes targeting float16
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT
    
    # 4. Fix Constant nodes with float16 tensors
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT16:
                    arr = numpy_helper.to_array(attr.t).astype(np.float32)
                    new_t = numpy_helper.from_array(arr)
                    attr.t.CopyFrom(new_t)

    onnx.save(model, fp32_path)
    
    fp32_size = os.path.getsize(fp32_path) / (1024*1024)
    print(f"  Saved FP32 encoder: {fp32_size:.1f} MB")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")

    # 1. Quantize span_rep.onnx (FP32 version exists)
    print("=== span_rep ===")
    fp32_span = INPUT_DIR / "span_rep.onnx"
    if fp32_span.exists():
        t0 = time.time()
        quantize_fp32_to_int8(str(fp32_span), str(OUTPUT_DIR / "span_rep_int8.onnx"))
        print(f"  Done in {time.time()-t0:.1f}s\n")
    else:
        print("  SKIP: span_rep.onnx not found\n")

    # 2. Quantize count_embed.onnx (FP32 version exists)
    print("=== count_embed ===")
    fp32_count = INPUT_DIR / "count_embed.onnx"
    if fp32_count.exists():
        t0 = time.time()
        quantize_fp32_to_int8(str(fp32_count), str(OUTPUT_DIR / "count_embed_int8.onnx"))
        print(f"  Done in {time.time()-t0:.1f}s\n")
    else:
        print("  SKIP: count_embed.onnx not found\n")

    # 3. Encoder: FP16 -> FP32 -> INT8
    print("=== encoder ===")
    fp16_encoder = INPUT_DIR / "encoder_fp16.onnx"
    if fp16_encoder.exists():
        t0 = time.time()
        tmp_fp32 = str(OUTPUT_DIR / "encoder_fp32_tmp.onnx")
        convert_encoder_fp16_to_fp32(str(fp16_encoder), tmp_fp32)
        quantize_fp32_to_int8(tmp_fp32, str(OUTPUT_DIR / "encoder_int8.onnx"))
        # Cleanup temp
        if os.path.exists(tmp_fp32):
            os.remove(tmp_fp32)
        print(f"  Done in {time.time()-t0:.1f}s\n")
    else:
        print("  SKIP: encoder_fp16.onnx not found\n")

    print("INT8 models saved to:", OUTPUT_DIR)
    print("\nFinal sizes:")
    for f in sorted(OUTPUT_DIR.glob("*_int8.onnx")):
        print(f"  {f.name}: {os.path.getsize(f)/(1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
