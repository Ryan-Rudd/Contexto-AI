import numpy as np
import json
import os
import random
from sklearn.decomposition import PCA

class AISolver:
    def __init__(self, vocab_path="vocab.json", embeddings_path="embeddings.bin"):
        self.vocab = []
        self.embeddings = None
        self.word_to_idx = {}
        self.rank_to_sim = None
        self.pca_3d = None
        self.projected_embeddings = None
        
        self.load_model(vocab_path, embeddings_path)
        self.load_training_data()
        self.reset_session()

    def load_model(self, vocab_path, embeddings_path):
        print("Loading AI model...")
        with open(vocab_path) as f:
            self.vocab = json.load(f)
            
        raw = np.fromfile(embeddings_path, dtype=np.float32)
        dim = raw.shape[0] // len(self.vocab)
        self.embeddings = raw.reshape(len(self.vocab), dim)
        
        # Normalize
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings /= norms
        
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        
        # Compute PCA for visualization (3 components) once
        # This might be slow for 40k words? PCA on 40k x 100 is fast enough (~1s).
        print("Computing 3D projection for visualization...")
        self.pca_3d = PCA(n_components=3)
        self.projected_embeddings = self.pca_3d.fit_transform(self.embeddings)
        print("Model loaded.")

    def load_training_data(self):
        # Load or initialize calibration curve
        if os.path.exists("ai_calibration.json"):
            with open("ai_calibration.json") as f:
                data = json.load(f)
                self.rank_to_sim = np.array(data["rank_to_sim"])
                print("Loaded existing calibration data.")
        else:
            print("Calibrating new model...")
            self.calibrate()

    def save_training_data(self):
        with open("ai_calibration.json", "w") as f:
            json.dump({
                "rank_to_sim": self.rank_to_sim.tolist()
            }, f)
        print("Calibration data saved.")

    def calibrate(self):
        sample_size = 100
        vocab_size = len(self.vocab)
        sample_idxs = np.random.choice(vocab_size, sample_size, replace=False)
        curve_accum = np.zeros(vocab_size)

        for idx in sample_idxs:
            sims = self.embeddings @ self.embeddings[idx]
            sims_sorted = np.sort(sims)[::-1]
            curve_accum += sims_sorted

        self.rank_to_sim = curve_accum / sample_size
        self.save_training_data()

    def reset_session(self):
        self.candidate_scores = np.zeros(len(self.vocab))
        self.guessed_indices = set()
        self.history = [] # (word, rank, idx)
        self.first_guess = True

    def update_calibration(self, target_idx):
        # After a game, we can use the "real" target interactions to refine our curve
        # Simply weighted average slightly towards the new data
        real_sims = self.embeddings @ self.embeddings[target_idx]
        real_sims_sorted = np.sort(real_sims)[::-1]
        
        # Learning rate
        alpha = 0.1
        self.rank_to_sim = (1 - alpha) * self.rank_to_sim + alpha * real_sims_sorted
        self.save_training_data()

    def get_next_guess(self):
        if self.first_guess:
            self.first_guess = False
            guess_idx = random.randint(0, len(self.vocab) - 1)
            return guess_idx, "Initial random exploration."
            
        # Select best score
        mask = np.ones(len(self.vocab), dtype=bool)
        if self.guessed_indices:
            mask[list(self.guessed_indices)] = False
            
        valid_indices = np.where(mask)[0]
        # Add some randomness to top scores to avoid getting stuck? 
        # For now argmax is fine as score is deterministic based on history.
        best_local_idx = np.argmax(self.candidate_scores[mask])
        guess_idx = valid_indices[best_local_idx]
        
        reasoning = self._generate_reasoning(guess_idx)
        return guess_idx, reasoning

    def _generate_reasoning(self, guess_idx):
        if not self.history:
            return "First strategic guess."
            
        sorted_history = sorted(self.history, key=lambda x: x[1])
        best_support = None
        min_diff = 1e9

        for word, rank, idx in sorted_history[:3]:
            # What similarity did we EXPECT based on rank?
            expected_sim = self.rank_to_sim[rank-1]
            # What similarity do we HAVE between candidate and this history word?
            actual_sim = float(self.embeddings[guess_idx] @ self.embeddings[idx])
            
            diff = abs(expected_sim - actual_sim)
            if diff < min_diff:
                min_diff = diff
                best_support = (word, rank, actual_sim, expected_sim)
        
        if best_support:
            ref_word, ref_rank, act, exp = best_support
            return f"Fits constraint from '{ref_word}' (#{ref_rank}). (Sim: {act:.2f} vs Exp: {exp:.2f})"
            
        return "Fits the intersection of previous clues."

    def process_result(self, guess_idx, rank):
        self.guessed_indices.add(guess_idx)
        self.candidate_scores[guess_idx] = -1e9 # Eliminate
        
        word = self.vocab[guess_idx]
        self.history.append((word, rank, guess_idx))
        
        if rank == 1:
            self.update_calibration(guess_idx)
            return

        # Update scores
        target_sim = self.rank_to_sim[rank - 1]
        current_sims = self.embeddings @ self.embeddings[guess_idx]
        delta = current_sims - target_sim
        
        # Update function: Score -= error^2
        self.candidate_scores -= (delta ** 2) * 10.0

    def get_3d_coordinates(self, indices):
        # Return list of [x, y, z] for given indices
        return self.projected_embeddings[indices].tolist()
        
    def get_vocab_word(self, idx):
        return self.vocab[idx]
