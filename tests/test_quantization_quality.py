import unittest
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from optimum.onnxruntime import ORTModelForFeatureExtraction

class TestQuantizationQuality(unittest.TestCase):
    def setUp(self):
        self.model_id = "intfloat/multilingual-e5-small"
        self.quantized_path = "models_quantized/multilingual-e5-small/quantized"
        
        # Load PT Model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.pt_model = AutoModel.from_pretrained(self.model_id)
        self.pt_model.eval()
        
        # Load ONNX INT8 Model
        self.onnx_model = ORTModelForFeatureExtraction.from_pretrained(self.quantized_path)

    def test_cosine_similarity(self):
        sentences = [
            "Hello world",
            "Artificial Intelligence optimization",
            "Complex query about semantic search systems."
        ]
        
        # 1. PT Output
        pt_inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            pt_out = self.pt_model(**pt_inputs)
        
        # PT Pooling
        pt_last = pt_out.last_hidden_state
        pt_mask = pt_inputs.attention_mask
        exp_mask = pt_mask.unsqueeze(-1).expand(pt_last.size()).float()
        pt_sum = torch.sum(pt_last * exp_mask, 1)
        pt_clamp = torch.clamp(exp_mask.sum(1), min=1e-9)
        pt_emb = pt_sum / pt_clamp
        pt_emb = torch.nn.functional.normalize(pt_emb, p=2, dim=1)
        pt_vecs = pt_emb.numpy()
        
        # 2. ONNX Output (using same inputs logic, but ONNX execution)
        onnx_inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        onnx_out = self.onnx_model(**onnx_inputs)
        
        # ONNX Pooling (Manual PT for comparison, to isolate Quantization noise)
        onnx_last = onnx_out.last_hidden_state
        # Note: onnx_last is usually torch tensor if return_tensors='pt' was passed to tokenizer 
        # BUT ORTModel might return numpy or torch depending on config.
        # Optimum ORT usually returns valid PyTorch tensors if inputs were PT!
        
        onnx_sum = torch.sum(onnx_last * exp_mask, 1)
        onnx_emb = onnx_sum / pt_clamp
        onnx_emb = torch.nn.functional.normalize(onnx_emb, p=2, dim=1)
        onnx_vecs = onnx_emb.numpy()
        
        # 3. Compare Cosine Similarity
        for i in range(len(sentences)):
            cos_sim = np.dot(pt_vecs[i], onnx_vecs[i]) / (np.linalg.norm(pt_vecs[i]) * np.linalg.norm(onnx_vecs[i]))
            print(f"Sentence {i} Cosine Sim: {cos_sim:.5f}")
            self.assertTrue(cos_sim > 0.98, f"Quantization degradation too high: {cos_sim}")

if __name__ == "__main__":
    unittest.main()
