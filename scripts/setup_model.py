import sys
import subprocess
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import numpy as np
except ImportError:
    print("Installing numpy...")
    install("numpy")
    import numpy as np

try:
    import gensim.downloader as api
except ImportError:
    print("Installing gensim...")
    try:
        install("gensim")
    except Exception as e:
        print(f"Error installing gensim: {e}")
        sys.exit(1)
    import gensim.downloader as api

import json

def setup_model():
    print("Downloading GloVe model (glove-wiki-gigaword-100)...")
    # Using 'glove-wiki-gigaword-100' (128MB) - much better semantic relationships
    model = api.load("glove-wiki-gigaword-100")
    
    print("Model loaded. Saving for web use...")
    
    vocab = []
    vector_list = []
    
    # Extract vocabulary and vectors
    # Limit to top 40,000 words for better synonym coverage
    limit = 40000
    count = 0
    
    # Get the vectors
    # model.key_to_index is a dict {word: index}
    # model.vectors is the numpy array
    
    # We iterate through the MOST COMMON words first (index 0 is most common)
    for word in model.index_to_key:
        if count >= limit:
            break
            
        # Filter mostly alphanumeric words to keep the game clean
        if word.isalnum() and len(word) > 2:
            vocab.append(word)
            vector_list.append(model[word])
            count += 1
            
    # Save vocabulary
    with open("vocab.json", "w") as f:
        json.dump(vocab, f)
        
    # Save vectors as binary float32
    # Javascript can read this easily with Float32Array
    vec_array = np.array(vector_list, dtype=np.float32)
    
    # Flatten array for binary storage
    flat_array = vec_array.flatten()
    
    # Save raw bytes
    flat_array.tofile("embeddings.bin")
    
    print(f"Done! Saved {len(vocab)} words to vocab.json and embeddings.bin")
    print("Files ready for index.html")

if __name__ == "__main__":
    setup_model()
