# AI Plagiarism Detector

A full-stack plagiarism detection tool built with FastAPI, FAISS, and a modern frontend UI.

## Features

- Paste text and check for plagiarism against a local document corpus
- Upload files for checking: `.pdf`, `.docx`, `.doc`, `.txt`
- Displays overall plagiarism score and matched source documents
- Highlights the input sentences most likely triggering plagiarism
- Uses BERT-based semantic embeddings and FAISS similarity search

## Repository Structure

- `backend/`
  - `main.py` - FastAPI backend service and endpoints
  - `app/`
    - `engine.py` - main plagiarism detection engine
    - `data_utils.py` - document loading and text cleaning
    - `file_utils.py` - PDF/DOCX/TXT extraction utilities
    - `model_utils.py` - embedding generation and caching
    - `search_utils.py` - FAISS index creation and similarity search
  - `data/` - sample documents used for plagiarism detection
- `frontend/`
  - `index.html` - UI page
  - `script.js` - frontend logic and API integration
  - `style.css` - styling and layout
- `requirements.txt` - Python dependencies

## Setup

1. Create and activate the Python virtual environment:

```powershell
cd d:\hackathon\plagiarism
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start the backend server:

```powershell
python backend/main.py
```

4. Open the frontend page in your browser:

- `frontend/index.html`

> Note: The backend runs at `http://localhost:8000` and the frontend communicates with that endpoint.

## Usage

- Paste text into the input tab and click **Check Plagiarism**
- Or switch to the **Upload File** tab and upload a supported document
- Review the overall score, matched sources, and triggering sentences

## Notes

- The app uses `sentence-transformers/all-MiniLM-L6-v2` for embedding generation
- Cached embeddings and FAISS index files are stored in the backend directory
- Add or update sample documents in `backend/data/` to expand the detection corpus
