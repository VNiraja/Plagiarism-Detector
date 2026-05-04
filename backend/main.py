from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.engine import PlagiarismEngine
from app.file_utils import extract_text_from_file
import uvicorn
import os
import tempfile
from fastapi import File, UploadFile

app = FastAPI(title="AI Plagiarism Detector API")

# Enable CORS (Cross-Origin Resource Sharing)
# This allows our frontend (HTML file) to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, you would restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the engine
engine = PlagiarismEngine()

@app.on_event("startup")
async def startup_event():
    # This runs when the server starts
    engine.initialize()

class PlagiarismRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Plagiarism Detector API"}

@app.post("/check-plagiarism")
async def check_plagiarism(request: PlagiarismRequest):
    if not request.text or len(request.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text too short to check.")
    
    try:
        result_data = engine.check_text(request.text)
        results = result_data.get("results", [])
        triggering_parts = result_data.get("triggering_parts", [])
        
        # Calculate an overall plagiarism score (highest single match)
        overall_score = max([r['similarity_score'] for r in results]) if results else 0
        
        return {
            "overall_score": overall_score,
            "matches": results,
            "triggering_parts": triggering_parts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check-plagiarism-file")
async def check_plagiarism_file(file: UploadFile = File(...)):
    """
    Handles file uploads (PDF, DOCX, DOC, TXT) and checks for plagiarism.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    
    # Validate file extension
    valid_extensions = ['.pdf', '.docx', '.doc', '.txt']
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in valid_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {', '.join(valid_extensions)}")
    
    # Validate file size (10MB max)
    max_size = 10 * 1024 * 1024
    
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file_path = tmp_file.name
            
            # Read file content
            content = await file.read()
            
            # Check size
            if len(content) > max_size:
                raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
            
            tmp_file.write(content)
        
        # Extract text from file
        extracted_text = extract_text_from_file(tmp_file_path)
        
        # Check minimum length
        if len(extracted_text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Extracted text is too short to check.")
        
        # Run plagiarism check
        result_data = engine.check_text(extracted_text)
        results = result_data.get("results", [])
        triggering_parts = result_data.get("triggering_parts", [])
        
        # Calculate overall score
        overall_score = max([r['similarity_score'] for r in results]) if results else 0
        
        return {
            "overall_score": overall_score,
            "matches": results,
            "triggering_parts": triggering_parts,
            "filename": file.filename,
            "extracted_text": extracted_text[:500]  # Return first 500 chars as preview
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
