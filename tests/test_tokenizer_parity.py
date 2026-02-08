import numpy as np
from transformers import AutoTokenizer
from tokenizers import Tokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_parity")

def test_tokenizer_parity():
    # Path to quantized model directory (contains tokenizer.json)
    model_path = "models_quantized/multilingual-e5-small/quantized"
    
    logger.info(f"Loading reference tokenizer from {model_path}...")
    try:
        ref_tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        logger.error(f"Failed to load Reference AutoTokenizer: {e}")
        return

    logger.info(f"Loading candidate tokenizer from {model_path}/tokenizer.json...")
    try:
        cand_tokenizer = Tokenizer.from_file(f"{model_path}/tokenizer.json")
    except Exception as e:
        logger.error(f"Failed to load mult-file tokenizer: {e}")
        return

    # Test Sentences
    test_sentences = [
        "Hello world",
        "This is a longer sentence to test truncation.",
        "Edge case",
    ]

    # Configuration for e5-small (Usually max_length=512)
    MAX_LEN = 512
    
    # Configure Candidate
    # Note: Enable padding/truncation to match AutoTokenizer behavior
    cand_tokenizer.enable_truncation(max_length=MAX_LEN)
    cand_tokenizer.enable_padding(length=MAX_LEN, pad_id=ref_tokenizer.pad_token_id)
    # Check if we need to set specific pad_token explicitly? 
    # AutoTokenizer usually handles this. Let's see if from_file picks it up.

    success = True
    for text in test_sentences:
        logger.info(f"Testing text: '{text}'")
        
        # Reference Encoding
        ref_enc = ref_tokenizer(text, padding="max_length", max_length=MAX_LEN, truncation=True, return_tensors="np")
        ref_ids = ref_enc["input_ids"][0]
        ref_att = ref_enc["attention_mask"][0]
        
        # Candidate Encoding
        cand_enc = cand_tokenizer.encode(text)
        cand_ids = np.array(cand_enc.ids)
        cand_att = np.array(cand_enc.attention_mask)
        
        # Compare
        if not np.array_equal(ref_ids, cand_ids):
            logger.error("Mismatch in Input IDs!")
            logger.error(f"Ref:  {ref_ids[:10]}... (Len: {len(ref_ids)})")
            logger.error(f"Cand: {cand_ids[:10]}... (Len: {len(cand_ids)})")
            
            # Check specifically for CLS/SEP
            # BERT CLS=101, SEP=102? Check tokenizer.
            cls_id = ref_tokenizer.cls_token_id
            sep_id = ref_tokenizer.sep_token_id
            logger.info(f"Expected CLS: {cls_id}, SEP: {sep_id}")
            
            if ref_ids[0] == cls_id and cand_ids[0] != cls_id:
                logger.error("CRITICAL: Candidate missing CLS token at start!")
            
            success = False
        else:
            logger.info("Input IDs Match ✅")

        if not np.array_equal(ref_att, cand_att):
            logger.error("Mismatch in Attention Mask!")
            success = False
        else:
            logger.info("Attention Mask Match ✅")
            
    if success:
        logger.info("\n>>> ALL CHECKS PASSED. Tokenizers are equivalent. <<<")
    else:
        logger.error("\n>>> CHECKS FAILED. Manual Post-Processing Required. <<<")
        exit(1)

if __name__ == "__main__":
    test_tokenizer_parity()
