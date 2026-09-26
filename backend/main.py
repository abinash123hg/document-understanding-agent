"""Document Understanding Agent - FastAPI application."""
import logging, uuid
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend import llm, processor, retriever, storage
from backend.config import settings
logging.basicConfig(level=logging.INFO); logger=logging.getLogger(__name__)
app=FastAPI(title="Document Understanding Agent",version="1.2.0",description="Local document Q&A with strict document-scoped retrieval.")
app.add_middleware(CORSMiddleware,allow_origins=[x.strip() for x in settings.cors_origins.split(",") if x.strip()],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
ALLOWED_EXTENSIONS={".pdf",".txt",".docx",".md",".png",".jpg",".jpeg"}
FALLBACK_ANSWER="I could not find the answer in this document."
class ChatRequest(BaseModel): question:str; document_name:Optional[str]=None
class Source(BaseModel): filename:str; chunk_index:int; score:float; text:str
class ChatResponse(BaseModel): answer:str; sources:List[Source]=Field(default_factory=list); document_name:Optional[str]=None
@app.get("/")
def root(): return {"status":"running","model":settings.llm_model,"mode":"document-only"}
@app.get("/health")
def health():
    ollama=llm.is_ollama_ready()
    return {"backend":"ok","ollama":ollama,"model":settings.llm_model,"model_available":ollama and llm.is_model_available(),"chunk_count":len(storage.get_chunks())}
@app.post("/upload")
async def upload(file:UploadFile=File(...)):
    if not file.filename: raise HTTPException(400,"No filename provided")
    ext=Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS: raise HTTPException(400,f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    max_bytes=settings.max_upload_size_mb*1024*1024; temp_path=settings.uploads_dir/f"{uuid.uuid4()}{ext}"; written=0
    try:
        with open(temp_path,"wb") as output:
            while True:
                chunk=await file.read(1024*1024)
                if not chunk: break
                written+=len(chunk)
                if written>max_bytes: raise HTTPException(413,f"File is too large. Maximum size is {settings.max_upload_size_mb} MB.")
                output.write(chunk)
        result=processor.process_upload(temp_path,file.filename)
        if result["status"]=="FAILED": raise HTTPException(422,result.get("error","Could not extract text"))
        return {"filename":file.filename,"document_name":file.filename,"status":"INDEXED","extraction_method":result.get("extraction_method"),"chunk_count":result.get("chunk_count",0)}
    except HTTPException: raise
    except Exception as exc: logger.exception("Upload processing failed"); raise HTTPException(500,f"Processing failed: {exc}")
    finally: temp_path.unlink(missing_ok=True)
@app.post("/documents/upload")
async def documents_upload(file:UploadFile=File(...)): return await upload(file)
@app.post("/chat",response_model=ChatResponse)
def chat(request:ChatRequest):
    question=request.question.strip()
    if not question: raise HTTPException(400,"Question cannot be empty")
    if not request.document_name: raise HTTPException(400,"Select a document before asking a question")
    sources=retriever.retrieve(question,settings.top_k,request.document_name)
    if not sources: return ChatResponse(answer=FALLBACK_ANSWER,sources=[],document_name=request.document_name)
    try: answer=llm.generate_answer(question,sources)
    except RuntimeError as exc: logger.error("LLM generation failed: %s",exc); raise HTTPException(503,str(exc))
    except Exception: logger.exception("Unexpected LLM generation failure"); raise HTTPException(503,"The local language model is unavailable. Please try again.")
    return ChatResponse(answer=answer,sources=[Source(filename=str(s.get("filename","")),chunk_index=int(s.get("chunk_index",0)),score=float(s.get("score",0)),text=str(s.get("text",""))[:300]) for s in sources],document_name=request.document_name)
@app.delete("/documents/{document_name}")
def delete_document(document_name:str): storage.clear_document(document_name); return {"status":"deleted","document_name":document_name}
