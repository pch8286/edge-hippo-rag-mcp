import sys
import os
import logging

# Add project root so the preferred seahorse package is importable.
sys.path.append(os.getcwd())

from seahorse.extraction import EntityExtractor

logging.basicConfig(level=logging.INFO)

def test_wrapper():
    print("Initializing EntityExtractor...")
    extractor = EntityExtractor()
    
    # Force load (it usually lazy loads on first predict, or we can trigger it)
    # Our EntityExtractor usually loads on first call.
    
    text = "The launch of the iPhone by Apple in 2007 changed technology forever."
    print(f"Extracting from: '{text}'")
    
    try:
        entities = extractor.extract_entities(text) # This is async wrapper?
        # Wait, extract_entities is async in the new code!
        # _extract_sync is the synchronous one.
        # But we can call _extract_sync directly for testing if we want simpler debug.
        
        # Let's use _extract_sync to avoid asyncio setup in simple script
        print("Calling _extract_sync...")
        entities = extractor._extract_sync(text)
        
        print(f"Found {len(entities)} entities:")
        for ent in entities:
            print(f" - {ent['label']}: {ent['text']} ({ent['score']:.4f})")
            
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_wrapper()
    sys.exit(0 if success else 1)
