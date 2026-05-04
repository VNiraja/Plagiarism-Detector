import faiss
import numpy as np
import os

def create_faiss_index(embeddings):
    """
    Creates a FAISS index from a set of embeddings.
    """
    # Get the dimension of the embeddings (for MiniLM it's 384)
    dimension = embeddings.shape[1]
    
    # We use IndexFlatL2 for exact search (best for smaller datasets like ours)
    # For massive datasets, we would use more complex indices
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to the index
    # FAISS requires float32 numpy arrays
    index.add(embeddings.astype('float32'))
    
    return index

def save_index(index, file_path):
    """
    Saves the FAISS index to a file.
    """
    faiss.write_index(index, file_path)

def load_index(file_path):
    """
    Loads the FAISS index from a file.
    """
    return faiss.read_index(file_path)

def search_similarity(query_embedding, index, k=5):
    """
    Searches for the top K most similar items in the index.
    Returns distances and indices.
    """
    # Ensure query embedding is 2D for FAISS
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    distances, indices = index.search(query_embedding.astype('float32'), k)
    return distances, indices

if __name__ == "__main__":
    # Test
    from model_utils import load_embeddings
    
    emb_path = os.path.join("backend", "embeddings.pkl")
    if os.path.exists(emb_path):
        embeddings = load_embeddings(emb_path)
        index = create_faiss_index(embeddings)
        print(f"Index created with {index.ntotal} vectors.")
        
        # Test search with the first embedding itself
        dist, idx = search_similarity(embeddings[0], index)
        print(f"Top match index: {idx[0][0]}, Distance: {dist[0][0]}")
    else:
        print("Embeddings not found. Run model_utils.py first.")
