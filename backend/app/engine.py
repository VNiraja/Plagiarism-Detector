import os
import numpy as np
from app.data_utils import load_documents, clean_text
from app.model_utils import generate_embeddings, save_embeddings, load_embeddings, model
from app.search_utils import create_faiss_index, search_similarity, save_index, load_index
import re

class PlagiarismEngine:
    def __init__(self, data_dir="backend/data"):
        self.data_dir = data_dir
        self.docs = []
        self.index = None
        self.embeddings = None
        
        # Paths for caching
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.embeddings_path = os.path.join(base_dir, "embeddings.pkl")
        self.index_path = os.path.join(base_dir, "faiss.index")
        
    def initialize(self, force_refresh=False):
        """
        Initializes the system by loading docs, generating embeddings, and building the index.
        """
        print("Initializing Plagiarism Engine...")
        self.docs = load_documents(self.data_dir)
        
        if not force_refresh and os.path.exists(self.embeddings_path) and os.path.exists(self.index_path):
            print("Loading cached embeddings and index...")
            self.embeddings = load_embeddings(self.embeddings_path)
            self.index = load_index(self.index_path)
        else:
            print("Generating new embeddings (this may take a moment)...")
            contents = [d['cleaned_content'] for d in self.docs]
            self.embeddings = generate_embeddings(contents)
            save_embeddings(self.embeddings, self.embeddings_path)
            
            print("Building FAISS index...")
            self.index = create_faiss_index(self.embeddings)
            save_index(self.index, self.index_path)
        
        print("System Ready.")

    def check_text(self, input_text, top_k=5):
        """
        Main logic to check user text against the database.
        Returns results with triggering parts identified.
        """
        cleaned_input = clean_text(input_text)
        query_embedding = generate_embeddings([cleaned_input])
        
        distances, indices = search_similarity(query_embedding, self.index, k=top_k)
        
        # Analyze which parts trigger plagiarism
        triggering_parts = self.identify_triggering_parts(input_text)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            
            similarity = 1 / (1 + float(dist))
            percentage = float(round(similarity * 100, 2))
            
            # Step 7: Highlight matching sentences
            source_content = self.docs[idx]["content"]
            highlights = self.get_highlights(input_text, source_content)
            
            results.append({
                "filename": self.docs[idx]["filename"],
                "content": source_content,
                "similarity_score": percentage,
                "highlights": highlights
            })
            
        return {
            "results": results,
            "triggering_parts": triggering_parts
        }

    def get_highlights(self, input_text, source_text):
        """
        Compares input and source sentence-by-sentence to find matching parts.
        """
        # Simple sentence splitter
        input_sentences = re.split(r'(?<=[.!?]) +', input_text)
        source_sentences = re.split(r'(?<=[.!?]) +', source_text)
        
        # Clean sentences for better matching
        clean_input = [clean_text(s) for s in input_sentences if len(s.strip()) > 5]
        clean_source = [clean_text(s) for s in source_sentences if len(s.strip()) > 5]
        
        if not clean_input or not clean_source:
            return []

        # Generate embeddings for all sentences
        input_embs = generate_embeddings(clean_input)
        source_embs = generate_embeddings(clean_source)
        
        matches = []
        
        # Compare every input sentence with every source sentence
        # Using dot product (cosine similarity) for faster comparison here
        for i, i_emb in enumerate(input_embs):
            for j, s_emb in enumerate(source_embs):
                # Cosine similarity
                sim = np.dot(i_emb, s_emb) / (np.linalg.norm(i_emb) * np.linalg.norm(s_emb))
                
                if sim > 0.85: # Threshold for "highly similar"
                    matches.append({
                        "input_sentence": input_sentences[i],
                        "source_sentence": source_sentences[j],
                        "score": round(float(sim), 2)
                    })
                    break # Move to next input sentence once a match is found
                    
        return matches

    def identify_triggering_parts(self, input_text, threshold=0.65):
        """
        Identifies which sentences/parts of the input text have high similarity to database.
        Returns sentences ranked by their plagiarism risk.
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?]) +', input_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if not sentences:
            return []
        
        # Generate embeddings for input sentences
        sentence_embeddings = generate_embeddings(sentences)
        
        triggering_sentences = []
        
        # Compare each sentence with all documents in database using cosine similarity
        for sent_idx, sent_emb in enumerate(sentence_embeddings):
            max_similarity = 0.0
            
            # Compare against all document embeddings
            for doc_emb in self.embeddings:
                # Cosine similarity: dot product / (norm1 * norm2)
                dot_product = np.dot(sent_emb, doc_emb)
                norm_sent = np.linalg.norm(sent_emb)
                norm_doc = np.linalg.norm(doc_emb)
                
                if norm_sent > 0 and norm_doc > 0:
                    sim = dot_product / (norm_sent * norm_doc)
                    max_similarity = max(max_similarity, float(sim))
            
            # If similarity is above threshold, it's a triggering part
            if max_similarity > threshold:
                triggering_sentences.append({
                    "sentence": sentences[sent_idx],
                    "plagiarism_risk": float(round(max_similarity * 100, 2)),
                    "position": sent_idx
                })
        
        # Sort by plagiarism risk (descending)
        triggering_sentences.sort(key=lambda x: x['plagiarism_risk'], reverse=True)
        
        return triggering_sentences

if __name__ == "__main__":
    # Test script
    engine = PlagiarismEngine()
    engine.initialize()
    
    test_text = "Artificial intelligence involves building machines that can think like humans."
    matches = engine.check_text(test_text)
    
    print(f"\nResults for: '{test_text}'")
    for m in matches:
        print(f"Match: {m['filename']} | Score: {m['similarity_score']}%")
