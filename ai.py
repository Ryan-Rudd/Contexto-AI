import json
import numpy as np
import requests
import random
import time

API = "http://localhost:8080/api/guess"

print("Loading model for AI solver...")
with open("vocab.json") as f:
    vocab = json.load(f)

raw = np.fromfile("embeddings.bin", dtype=np.float32)
DIM = raw.shape[0] // len(vocab)
embeddings = raw.reshape(len(vocab), DIM)

# Normalize for cosine similarity
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1.0
embeddings /= norms

vocab_size = len(vocab)
word_to_idx = {w: i for i, w in enumerate(vocab)}

# --- Calibration ---
# We need to map "Rank" to "Expected Cosine Similarity".
# We'll sample a few words to build this curve.
print("Calibrating Rank-Similarity curve...")
sample_size = 50
sample_idxs = np.random.choice(vocab_size, sample_size, replace=False)
curve_accum = np.zeros(vocab_size)

for idx in sample_idxs:
    # Dot product with all others
    sims = embeddings @ embeddings[idx]
    # Sort descending to get sim at rank 0, 1, 2...
    # We want rank 1 (self) to be index 0
    sims_sorted = np.sort(sims)[::-1]
    curve_accum += sims_sorted

rank_to_sim = curve_accum / sample_size
print("Calibration done.")

# --- Solver State ---
candidate_scores = np.zeros(vocab_size)
guessed_indices = set()
history = [] # List of (word, rank, idx)

def get_reasoning(current_idx):
    if not history:
        return "Starting with a random exploratory guess."
    
    # improved reasoning: find which previous guess supports this choice the most
    # We want to find a previous guess G where dist(current, G) approx expected_dist(target, G)
    
    best_support = None
    min_diff = 1e9
    
    # Sort history by rank to prioritize good clues
    sorted_history = sorted(history, key=lambda x: x[1])
    
    # Check top 3 best clues
    for word, rank, idx in sorted_history[:3]:
        expected_sim = rank_to_sim[rank-1]
        actual_sim = float(embeddings[current_idx] @ embeddings[idx])
        
        diff = abs(expected_sim - actual_sim)
        if diff < min_diff:
            min_diff = diff
            best_support = (word, rank, actual_sim, expected_sim)
            
    if best_support:
        ref_word, ref_rank, act, exp = best_support
        return f"It fits the constraint from '{ref_word}' (#{ref_rank}). (Sim: {act:.3f} vs Exp: {exp:.3f})"
        
    return "It statistically fits the intersection of all previous clues."

def get_best_guess():
    mask = np.ones(vocab_size, dtype=bool)
    if guessed_indices:
        mask[list(guessed_indices)] = False
    
    valid_indices = np.where(mask)[0]
    best_local_idx = np.argmax(candidate_scores[mask])
    return valid_indices[best_local_idx]

# --- Main Loop ---
step = 0
first_guess = True

while True:
    step += 1
    
    if first_guess:
        guess_idx = random.randint(0, vocab_size - 1)
        reasoning = "Initial random guess."
        first_guess = False
    else:
        guess_idx = get_best_guess()
        reasoning = get_reasoning(guess_idx)
        
    word = vocab[guess_idx]
    guessed_indices.add(guess_idx)
    candidate_scores[guess_idx] = -1e9

    try:
        print(f"I am guessing '{word}' because: {reasoning}")
        res = requests.post(API, json={"word": word}).json()
        if "error" in res:
            print(f"Error: {res['error']}")
            continue
            
        rank = res["rank"]
        print(f"{step:02d}. {word:<15} → rank {rank:5d}\n")
        
        history.append((word, rank, guess_idx))

        if rank == 1:
            print("Solved!")
            break

        target_sim = rank_to_sim[rank - 1]
        current_sims = embeddings @ embeddings[guess_idx]
        delta = current_sims - target_sim
        candidate_scores -= (delta ** 2) * 10.0 
        
    except Exception as e:
        print(f"Loop error: {e}")
        break
