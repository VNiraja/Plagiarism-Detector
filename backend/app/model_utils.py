import os
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Load the model globally (it will be downloaded on first run)
# 'all-MiniLM-L6-v2' is a fast and accurate model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(texts):
    """
    Converts a list of strings into a list of numerical embeddings.
    """
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def save_embeddings(embeddings, file_path):
    """
    Saves embeddings to a pickle file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(file_path):
    """
    Loads embeddings from a pickle file.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    from data_utils import load_documents
    
    # Test path
    data_dir = os.path.join("backend", "data")
    docs = load_documents(data_dir)
    contents = [d['cleaned_content'] for d in docs]
    
    print("Generating embeddings...")
    embeddings = generate_embeddings(contents)
    print(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    # Save test
    save_path = os.path.join("backend", "embeddings.pkl")
    save_embeddings(embeddings, save_path)
    print(f"Saved embeddings to {save_path}")
