import os
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import random
from ai_solver import AISolver

app = Flask(__name__, static_folder='.')

# --- Models ---
vocab = []
embeddings = None
word_to_index = {}
target_word = ""
target_vector = None
target_similarities = None 
sorted_indices = None 

# AI Instance
ai_solver = None

def load_game_model():
    global vocab, embeddings, word_to_index
    print("Loading game model...")
    with open("vocab.json", "r") as f:
        vocab = json.load(f)
    word_to_index = {word: i for i, word in enumerate(vocab)}
    
    raw = np.fromfile("embeddings.bin", dtype=np.float32)
    dim = raw.shape[0] // len(vocab)
    embeddings = raw.reshape(len(vocab), dim)
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    print("Game model loaded.")

def init_ai():
    global ai_solver
    if not ai_solver:
        ai_solver = AISolver()

def start_new_game():
    global target_word, target_vector, target_similarities, sorted_indices
    
    idx = random.randint(0, len(vocab) - 1)
    target_word = vocab[idx]
    target_vector = embeddings[idx]
    
    print(f"New game started! Target: {target_word}")
    
    # Pre-calc distances for the user
    target_similarities = np.dot(embeddings, target_vector)
    sorted_indices = np.argsort(target_similarities)[::-1]
    
    # Reset AI for this new game
    if ai_solver:
        ai_solver.reset_session()

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/api/new_game', methods=['POST'])
def new_game():
    start_new_game()
    
    bg_points = []
    if ai_solver:
        # Sample 1000 random points for cosmic background
        try:
            total = len(ai_solver.vocab)
            indices = np.random.choice(total, 1000, replace=False)
            bg_points = ai_solver.projected_embeddings[indices].tolist()
        except Exception as e:
            print(f"Error generating background: {e}")

    return jsonify({
        "message": "New game started", 
        "debug_word": target_word,
        "background": bg_points
    })

@app.route('/api/guess', methods=['POST'])
def guess():
    global ai_solver
    if not target_word:
        start_new_game()
        
    data = request.json
    guess_word = data.get('word', '').lower().strip()
    
    if guess_word not in word_to_index:
        return jsonify({"error": "Word not found in vocabulary"}), 404
        
    idx = word_to_index[guess_word]
    
    # Determine rank
    rank_idx = np.where(sorted_indices == idx)[0][0]
    rank = int(rank_idx) + 1
    
    # Update AI state if it exists, so it learns from user moves too
    if ai_solver:
        if guess_word in ai_solver.word_to_idx:
            ai_idx = ai_solver.word_to_idx[guess_word]
            ai_solver.process_result(ai_idx, rank)
            
    # Get 3D coords for graph
    coords = None
    target_coords = None
    if ai_solver:
        if guess_word in ai_solver.word_to_idx:
            ai_idx = ai_solver.word_to_idx[guess_word]
            coords = ai_solver.projected_embeddings[ai_idx].tolist()
            
        if rank == 1 and target_word in ai_solver.word_to_idx:
            t_idx = ai_solver.word_to_idx[target_word]
            target_coords = ai_solver.projected_embeddings[t_idx].tolist()

    return jsonify({
        "word": guess_word,
        "rank": rank,
        "distance": rank,
        "coords": coords,
        "target_coords": target_coords
    })

@app.route('/api/ai/guess', methods=['POST'])
def ai_guess():
    global ai_solver
    if not ai_solver:
        return jsonify({"error": "AI not initialized"}), 500
        
    # AI decides what to guess
    idx, reasoning = ai_solver.get_next_guess()
    word = ai_solver.vocab[idx]
    
    # Get rank (assuming game is running)
    if word not in word_to_index:
         return jsonify({"error": "AI word mismatch"}), 500
         
    game_idx = word_to_index[word]
    rank_idx = np.where(sorted_indices == game_idx)[0][0]
    rank = int(rank_idx) + 1
    
    # Update AI internal state with the result
    ai_solver.process_result(idx, rank)
    
    # Get 3D coords
    coords = ai_solver.projected_embeddings[idx].tolist()
    
    # Also need to add to frontend history
    return jsonify({
        "word": word,
        "reasoning": reasoning,
        "rank": rank,
        "distance": rank,
        "coords": coords
    })

if __name__ == '__main__':
    load_game_model()
    init_ai()
    start_new_game()
    app.run(host='0.0.0.0', port=8080, debug=True)
