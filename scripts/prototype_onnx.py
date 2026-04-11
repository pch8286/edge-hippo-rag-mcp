import numpy as np
import torch
from transformers import AutoTokenizer
import collections

# --- Config ---
MODEL_NAME = "fastino/gliner2-multi-v1"
MAX_WIDTH = 12

class MockSession:
    def __init__(self, output_shapes):
        self.output_shapes = output_shapes
    
    def run(self, output_names, input_feed):
        # Infer shapes from inputs somewhat dynamically
        input_ids = input_feed.get("input_ids", None)
        seq_len = input_ids.shape[1] if input_ids is not None else 0
        
        result = []
        if "last_hidden_state" in self.output_shapes: 
            res = np.random.randn(1, seq_len, 768).astype(np.float32)
            result.append(res)
            
        if "span_representations" in self.output_shapes:
            start_idx = input_feed.get("span_start_idx")
            if start_idx is not None:
                # Handle potentially different shapes if flattened or batched
                # Here we assume (Batch, NumSpans)
                num_spans = start_idx.shape[1] 
                res = np.random.randn(1, num_spans, 768).astype(np.float32)
                result.append(res)
                
        if "transformed_embeddings" in self.output_shapes:
            l_embeds = input_feed.get("label_embeddings")
            if l_embeds is not None:
                num_labels = l_embeds.shape[0]
                res = np.random.randn(num_labels, 768).astype(np.float32)
                result.append(res)
                
        return result

def extract_word_embeddings(token_embeds, words_mask, max_text_length):
    batch_idx, word_idx = np.where(words_mask > 0)
    target_word_idx = words_mask[batch_idx, word_idx] - 1
    
    words_embedding = np.zeros((1, max_text_length, 768), dtype=np.float32)
    words_embedding[batch_idx, target_word_idx] = token_embeds[batch_idx, word_idx]
    
    return words_embedding

def extract_prompt_features(token_embeds, input_ids, ent_token_id):
    # Find [ENT] tokens
    # Note: Using == comparison on numpy array
    mask = (input_ids == ent_token_id)
    batch_idx, token_idx = np.where(mask)
    
    # We want token at [ENT] position (assuming embed_ent_token=True)
    # If using [E] Label, and we want Label embedding, we'd add +1. 
    # But usually "Prompt Embeddings" ARE the [ENT] token embeddings after context.
    # GLiNER v1 uses [ENT]. v2 uses [E]?
    # Let's assume we extract the [E] token's contextual embedding.
    
    if len(batch_idx) == 0:
        return np.zeros((1, 0, 768), dtype=np.float32)
        
    prompt_embeds = token_embeds[batch_idx, token_idx]
    prompt_embeds = prompt_embeds.reshape(1, -1, 768)
    return prompt_embeds

class PureOnnxGLiNER:
    def __init__(self, model_name="fastino/gliner2-multi-v1"):
        print(f"Loading tokenizer: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # GLiNER 2 Token Logic
        # Based on config.json
        vocab = self.tokenizer.get_vocab()
        
        if "[E]" in vocab:
            self.ent_token = "[E]"
        elif "<<" in vocab: # <<ENT>>?
            self.ent_token = "<<ENT>>"
        else:
            self.ent_token = "[ENT]" # Fallback
            
        if "[SEP_TEXT]" in vocab:
            self.sep_token = "[SEP_TEXT]" 
        else:
            self.sep_token = "[SEP]"

        print(f"Using Tokens -> ENT: {self.ent_token}, SEP: {self.sep_token}")
        self.ent_token_id = vocab.get(self.ent_token, -1)
        
        if self.ent_token_id == -1:
            # Maybe it's a special token not in vocab dict but in additional_special_tokens_ids?
            # Or tokenizers handles it.
            # Let's rely on tokenizer.encode to check ID
            ids = self.tokenizer.encode(self.ent_token, add_special_tokens=False)
            if ids:
                self.ent_token_id = ids[0]
                print(f"Found ID for {self.ent_token}: {self.ent_token_id}")
            else:
                print(f"CRITICAL: Could not find ID for {self.ent_token}")

        self.encoder_session = MockSession(["last_hidden_state"])
        self.span_rep_session = MockSession(["span_representations"])
        self.count_embed_session = MockSession(["transformed_embeddings"])
        
    def prepare_span_idx(self, num_tokens, max_width=12):
        span_idx = []
        for i in range(num_tokens):
            for j in range(max_width):
                if i + j < num_tokens:
                    span_idx.append([i, i + j])
        return np.array(span_idx, dtype=np.int64)

    def prepare_input(self, text, labels):
        text_words = text.split() 
        num_text_words = len(text_words)
        
        prompt_words = []
        for label in labels:
            prompt_words.append(self.ent_token)
            prompt_words.append(label)
        prompt_words.append(self.sep_token)
        
        full_words = prompt_words + text_words
        
        encoding = self.tokenizer(
            full_words,
            is_split_into_words=True,
            return_tensors="np",
            padding=False, 
            truncation=True
        )
        
        input_ids = encoding["input_ids"] 
        attention_mask = encoding["attention_mask"]
        word_ids = encoding.word_ids() 
        
        words_mask = []
        prompt_len = len(prompt_words) 
        
        current_word_id = None
        for i, word_id in enumerate(word_ids):
            if word_id is None:
                words_mask.append(0)
            elif word_id < prompt_len:
                words_mask.append(0)
            else:
                if word_id != current_word_id:
                    relative_id = word_id - prompt_len + 1
                    words_mask.append(relative_id)
                else:
                    words_mask.append(0)
            current_word_id = word_id
            
        words_mask = np.array(words_mask, dtype=np.int64).reshape(1, -1)
        text_lengths = np.array([num_text_words], dtype=np.int64)
        span_idx = self.prepare_span_idx(num_text_words, MAX_WIDTH)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "words_mask": words_mask,
            "text_lengths": text_lengths,
            "span_idx": span_idx,
            "num_text_words": num_text_words
        }

    def predict(self, text, labels):
        print(f"\n--- Running Prediction with {len(labels)} labels ---")
        inputs = self.prepare_input(text, labels)
        
        # Check if ENT tokens are present in input_ids
        ent_count = np.sum(inputs["input_ids"] == self.ent_token_id)
        print(f"Input contains {ent_count} [ENT] tokens (Expected: {len(labels)})")
        
        # 1. Run Encoder
        enc_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        last_hidden_state = self.encoder_session.run(None, enc_inputs)[0]
        
        # 2. Extract Features
        prompts_embeds = extract_prompt_features(
            last_hidden_state, inputs["input_ids"], self.ent_token_id
        )
        print(f"Prompt Embeds Shape: {prompts_embeds.shape}") # Should be (1, NumLabels, 768)
        
        word_embeds = extract_word_embeddings(
            last_hidden_state, inputs["words_mask"], inputs["num_text_words"]
        )
        
        # 3. Transform Prompts
        p_in = prompts_embeds.reshape(-1, 768)
        # Skip if empty
        if p_in.shape[0] > 0:
            final_prompt_embeds = self.count_embed_session.run(None, {"label_embeddings": p_in})[0]
            final_prompt_embeds = final_prompt_embeds.reshape(1, -1, 768)
        else:
            final_prompt_embeds = prompts_embeds
            
        # 4. Span Rep
        span_idx = inputs["span_idx"] 
        span_start = span_idx[:, 0].reshape(1, -1)
        span_end = span_idx[:, 1].reshape(1, -1)
        
        span_rep_inputs = {
            "hidden_states": word_embeds,
            "span_start_idx": span_start,
            "span_end_idx": span_end
        }
        span_reps = self.span_rep_session.run(None, span_rep_inputs)[0]
        
        # 5. Scores
        scores = np.einsum("bsd,bld->bsl", span_reps, final_prompt_embeds)
        print(f"Final Scores Shape: {scores.shape}") 
        
        return scores

if __name__ == "__main__":
    model = PureOnnxGLiNER()
    model.predict("Bill Gates found Microsoft.", ["person", "org"])
