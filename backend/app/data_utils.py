import os
import re

def clean_text(text):
    """
    Cleans text for better similarity matching.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_documents(directory):
    """
    Loads all .txt files from the specified directory and returns a list of dictionaries.
    """
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                cleaned_content = clean_text(content)
                documents.append({
                    "filename": filename,
                    "content": content,
                    "cleaned_content": cleaned_content
                })
    return documents

if __name__ == "__main__":
    # Test loading
    data_dir = os.path.join("backend", "data")
    docs = load_documents(data_dir)
    print(f"Loaded {len(docs)} documents.")
    for d in docs[:3]:
        print(f"File: {d['filename']} | Content Snippet: {d['cleaned_content'][:50]}...")
