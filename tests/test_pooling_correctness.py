import unittest
import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModel
from typing import List

class TestPoolingCorrectness(unittest.TestCase):
    def setUp(self):
        self.model_id = "intfloat/multilingual-e5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
        self.model.eval()

    def test_numpy_mean_pooling_matches_pytorch(self):
        sentences = [
            "Hello world", 
            "This is a longer sentence to test pooling behavior correctly.", 
            "Short",
            "   Padding test   "
        ]
        
        # 1. PyTorch Reference
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs.attention_mask

        # PyTorch Reference Pooling Logic (E5 style: Mean Pooling)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pt_embeddings = sum_embeddings / sum_mask
        pt_embeddings = torch.nn.functional.normalize(pt_embeddings, p=2, dim=1)
        pt_vectors = pt_embeddings.numpy()

        # 2. NumPy Implementation
        # Simulate ONNX output (take raw hidden state from PT for isolation)
        np_last_hidden_state = last_hidden_state.numpy()
        np_attention_mask = attention_mask.numpy()
        
        def numpy_mean_pooling(last_hidden_state: np.ndarray, attention_mask: np.ndarray):
            # Shape: (batch, seq, dim)
            # Expand mask: (batch, seq) -> (batch, seq, 1) -> (batch, seq, dim)
            input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
            input_mask_expanded = np.broadcast_to(input_mask_expanded, last_hidden_state.shape).astype(np.float32)
            
            # Weighted Sum
            sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
            
            # Sum Mask
            sum_mask = np.sum(input_mask_expanded, axis=1)
            sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)
            
            # Divide
            embeddings = sum_embeddings / sum_mask
            
            # Normalize
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norm
            
            return embeddings

        np_vectors = numpy_mean_pooling(np_last_hidden_state, np_attention_mask)

        # 3. Comparison
        # Check L2 distance per vector
        for i in range(len(sentences)):
            diff = pt_vectors[i] - np_vectors[i]
            l2_dist = np.linalg.norm(diff)
            print(f"Sentence {i} L2 Dist: {l2_dist:.2e}")
            self.assertTrue(l2_dist < 1e-5, f"Sentence {i} deviation too high: {l2_dist}")

if __name__ == "__main__":
    unittest.main()
