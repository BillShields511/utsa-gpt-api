import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from gemini_agent import get_agent, load_or_embed
from firebase_script import load_documents


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str                # user's question from the React frontend

class ChatResponse(BaseModel):
    answer: str                  # Gemini's grounded answer
    sources: list[str]           # raw document snippets used as context

class HealthResponse(BaseModel):
    status: str                  # always "ok" when the server is running


# ---------------------------------------------------------------------------
# Lifespan — runs once at startup, tears down on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and documents once at server startup.

    Heavy operations (SentenceTransformer, CrossEncoder, Firestore stream,
    embedding) are offloaded to a thread pool so they don't block the event
    loop during startup. Results are stored on app.state so every request
    handler can reach them without re-initializing anything.
    """
    collection = os.getenv("FIRESTORE_COLLECTION", "game")
    template   = os.getenv("QUERY_REWRITE_TEMPLATE")   # optional .txt path

    # Loads SentenceTransformer + CrossEncoder — happens exactly once
    agent = get_agent(query_rewrite_template=template)

    # Stream documents from Firestore and compute/restore embeddings
    docs    = await asyncio.to_thread(load_documents, collection)
    texts   = [d.text for d in docs]
    vectors = await asyncio.to_thread(load_or_embed, agent, texts)

    app.state.agent   = agent
    app.state.docs    = texts
    app.state.vectors = vectors

    yield
    # Nothing to clean up — models live in memory until process exits


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="UTSA GPT API",
    description="RAG-powered chatbot grounded on user-supplied Firestore data.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    """Basic health check — confirms the server is running."""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(body: ChatRequest, request: Request):
    """Send a question and receive a grounded answer from Gemini.

    The answer is generated using only data retrieved from the Firestore
    collection loaded at startup. The response also includes the source
    document snippets that were used as context.
    """
    answer, sources = await request.app.state.agent.answer(
        body.question,
        request.app.state.docs,
        request.app.state.vectors,
    )
    return ChatResponse(answer=answer, sources=sources)
