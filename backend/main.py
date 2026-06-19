from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from processor import extract_text_from_pdf, chunk_legal_document
from vectorstore import LegalVectorStore, get_doc_id
from qa_chain import answer_question, generate_document_summary
import uvicorn

app = FastAPI(title="Indian Legal Q&A API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# initialise vector store
store = LegalVectorStore()

# in-memory chat history per document session
# in production this would be Redis
chat_histories: dict[str, list] = {}


class QuestionRequest(BaseModel):
    doc_id: str
    question: str
    include_history: bool = True


@app.get("/")
def root():
    return {
        "status": "running",
        "endpoints": {
            "POST /upload": "Upload a PDF document",
            "POST /question": "Ask a question about a document",
            "GET /summary/{doc_id}": "Get document summary",
            "GET /documents": "List all uploaded documents",
            "DELETE /document/{doc_id}": "Delete a document"
        }
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and index a PDF document.
    
    Pipeline:
    1. Read PDF bytes
    2. Extract text (with Indian legal PDF cleaning)
    3. Chunk with legal-aware splitter
    4. Embed chunks with sentence-transformers
    5. Store in ChromaDB
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are supported")
    
    file_bytes = await file.read()
    
    if len(file_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(400, "File too large. Maximum 10MB.")
    
    doc_id = get_doc_id(file.filename, file_bytes)
    
    # skip if already indexed
    if store.document_exists(doc_id):
        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "status": "already_indexed",
            "message": "Document was previously uploaded"
        }
    
    # process
    try:
        text = extract_text_from_pdf(file_bytes)
        
        if len(text.strip()) < 100:
            raise HTTPException(400, "Could not extract text from PDF")
        
        chunks = chunk_legal_document(text)
        
        if not chunks:
            raise HTTPException(400, "Could not split document into chunks")
        
        n_chunks = store.add_document(doc_id, chunks)
        
        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "status": "indexed",
            "chunks": n_chunks,
            "text_length": len(text),
            "message": f"Successfully indexed {n_chunks} chunks"
        }
        
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")


@app.post("/question")
async def ask_question(request: QuestionRequest):
    """
    Ask a question about an uploaded document.
    
    RAG pipeline:
    1. Embed the question
    2. Find top-5 most relevant chunks (semantic search)
    3. Pass question + chunks to LLM
    4. Return answer + source citations
    """
    if not store.document_exists(request.doc_id):
        raise HTTPException(404, "Document not found. Please upload it first.")
    
    # retrieve relevant chunks
    chunks = store.search(request.doc_id, request.question, k=5)
    
    if not chunks:
        raise HTTPException(500, "Could not retrieve relevant chunks")
    
    # get chat history
    history = []
    if request.include_history:
        history = chat_histories.get(request.doc_id, [])
    
    # generate answer
    result = answer_question(request.question, chunks, history)
    
    # update chat history
    if request.doc_id not in chat_histories:
        chat_histories[request.doc_id] = []
    
    chat_histories[request.doc_id].extend([
        {"role": "user", "content": request.question},
        {"role": "assistant", "content": result["answer"]}
    ])
    
    # keep history manageable
    if len(chat_histories[request.doc_id]) > 20:
        chat_histories[request.doc_id] = chat_histories[request.doc_id][-20:]
    
    return {
        "answer": result["answer"],
        "sources": [
            {
                "content": s["content"][:300] + "..." if len(s["content"]) > 300 else s["content"],
                "similarity": s["similarity"],
                "rank": s["rank"]
            }
            for s in result["sources"]
        ],
        "model": result["model"],
        "tokens_used": result["tokens_used"]
    }


@app.get("/summary/{doc_id}")
async def get_summary(doc_id: str):
    """Generate a structured summary of the document."""
    if not store.document_exists(doc_id):
        raise HTTPException(404, "Document not found")
    
    # get representative chunks for summary
    chunks = store.search(doc_id, "judgment decision held order conclusion", k=8)
    summary = generate_document_summary(chunks)
    
    return {"doc_id": doc_id, "summary": summary}


@app.get("/documents")
def list_documents():
    """List all uploaded documents."""
    return {"documents": store.list_documents()}


@app.delete("/document/{doc_id}")
def delete_document(doc_id: str):
    """Delete a document and its index."""
    try:
        store.client.delete_collection(f"legal_{doc_id}")
        if doc_id in chat_histories:
            del chat_histories[doc_id]
        return {"status": "deleted", "doc_id": doc_id}
    except:
        raise HTTPException(404, "Document not found")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)