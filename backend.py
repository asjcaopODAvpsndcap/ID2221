from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import uvicorn
import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.schema.runnable import RunnablePassthrough, RunnableConfig
from langchain.schema.output_parser import StrOutputParser

# --- Configuration ---
CHROMA_PERSIST_DIR = "./chroma_db_ollama"
OLLAMA_EMBEDDING_MODEL = "embeddinggemma:latest"
OLLAMA_LLM_MODEL = "deepseek-r1:7b"


# --- Pydantic Models  ---
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    temperature: Optional[float] = 0.3


class Document(BaseModel):
    pmid: str
    title: str
    content: str
    # score: Optional[float] = None # ChromaDB v0.4+ 不再默认返回 score


class AnswerResponse(BaseModel):
    answer: str
    references: List[Document]


class SystemStatus(BaseModel):
    status: str
    model: str
    embedding_model: str
    total_documents: int


# --- Global State ---
# 使用一个字典来存储全局状态，更清晰
app_state = {
    "vector_db": None,
    "llm": None,
    "rag_chain": None
}


# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("\n" + "=" * 60)
    print("🏥 Medical Literature QA System Starting...")
    print("=" * 60)
    try:
        if not os.path.exists(CHROMA_PERSIST_DIR):
            raise FileNotFoundError(
                f"Vector database not found at {CHROMA_PERSIST_DIR}. Please run build_index.py first.")

        print("🔄 Initializing embeddings...")
        embedding_function = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

        print("🔄 Loading vector database...")
        app_state["vector_db"] = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embedding_function)

        doc_count = app_state["vector_db"]._collection.count()
        if doc_count == 0:
            print("⚠️ Warning: Vector database is empty.")

        print("✅ System initialized successfully")
        print(f"   - LLM Model: {OLLAMA_LLM_MODEL}")
        print(f"   - Embedding: {OLLAMA_EMBEDDING_MODEL}")
        print(f"   - Documents: {doc_count}")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"❌ Initialization error: {e}")

    yield

    # Shutdown
    print("\n👋 Shutting down...")


# --- FastAPI App ---
app = FastAPI(title="Medical Literature QA API", version="1.0.0", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 核心修复 1: 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- API Endpoints ---

# ✅ 核心修复 2: 替换根路径("/")的端点，让它返回HTML文件
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the index.html file."""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")


@app.get("/api/status", response_model=SystemStatus)
async def get_status():
    if app_state["vector_db"] is None:
        raise HTTPException(status_code=503, detail="System is not initialized or failed to initialize.")

    return SystemStatus(
        status="online",
        model=OLLAMA_LLM_MODEL,
        embedding_model=OLLAMA_EMBEDDING_MODEL,
        total_documents=app_state["vector_db"]._collection.count()
    )


@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if app_state["vector_db"] is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        print(
            f"\n📝 Processing question: '{request.question[:50]}...' (Top K: {request.top_k}, Temp: {request.temperature})")

        # ✅ 优化 1: 动态配置 retriever 和 llm，而不是重新创建
        retriever = app_state["vector_db"].as_retriever(search_kwargs={"k": request.top_k})
        llm = Ollama(model=OLLAMA_LLM_MODEL, temperature=request.temperature)

        template = """You are a professional medical research assistant. Use the following context to answer the question. 
        If you don't know, just say that you don't know. Answer in Chinese.
        Context: {context}
        Question: {question}
        Answer (in Chinese):"""
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        # ✅ 优化 2: 使用 LangChain 的原生异步方法 .ainvoke()
        answer = await chain.ainvoke(request.question)

        # 异步获取参考文献
        docs = await retriever.ainvoke(request.question)

        references = [
            Document(
                pmid=doc.metadata.get('pmid', 'N/A'),
                title=doc.metadata.get('title', 'No Title'),
                content=doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            ) for doc in docs
        ]

        print(f"✅ Answer generated with {len(references)} references.")
        return AnswerResponse(answer=answer, references=references)

    except Exception as e:
        print(f"❌ Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Run Server ---
if __name__ == "__main__":
    # 移除 reload=True, uvicorn.run()本身不支持，应在命令行使用
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)