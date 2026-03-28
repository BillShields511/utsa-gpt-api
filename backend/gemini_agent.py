import os
import warnings
import logging
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

#file with embedded vectors to avoid re-embedding
CACHE_FILE = Path("vectors_cache.json")

#supress the really annoying warnings and logs from sentence transformers to keep the output clean in the CLI
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress the HF Hub unauthenticated warning
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_VERBOSITY"] = "error"

# Suppress the BertModel load report and progress bar
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

#load environment variables from .env file, which should include GEMINI_API_KEY for authentication with Gemini API
load_dotenv()

# Ensure requests go to the public Gemini API, not Vertex AI which was causing issues
for _key in ("GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI"):
    os.environ.pop(_key, None)

# GEMINI MODEL: we are using 2.5 flash but can be changed
_GEMINI_MODEL = "gemini-2.5-flash"
# embedding model for local vector search from huggingface sentence transformers
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
#prompt we feed to gemini when it gives the final answer
_RAG_PROMPT_TEMPLATE = """\
You are a helpful assistant that ONLY answers questions using the provided data.
Do NOT use any outside knowledge. If the answer is not in the data, say \
"I don't have that information."

Here is the data:
{context}

Question: {question}

Answer using ONLY the data above:"""

#prompt we feed to gemini to rewrite the user query for better retrieval.
_QUERY_REWRITE_TEMPLATE = """\
You are an expert at reformulating questions to improve document retrieval.
Given a user's question, rewrite it to be more specific and keyword-rich \
for searching an NFL schedule database.

The database contains: team names, game dates, days of the week, home/away designations.

Rules:
- Expand team nicknames to full names (e.g. "Cowboys" → "Dallas Cowboys")
- Make implicit info explicit (e.g. "this Sunday" → "Sunday")
- Keep it concise — one sentence
- Return ONLY the rewritten query, no explanation

Original question: {question}
Rewritten query:"""

# performs cosine similarity between the query vector and each document vector to get relevance scores for retrieval
def _cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Return cosine similarities between every row of *matrix* and *vector*."""
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    return np.dot(matrix, vector) / np.where(norms == 0, 1e-10, norms)

# Load vectors from cache if available, otherwise embed and cache them.
def load_or_embed(agent, contents):
    if CACHE_FILE.exists():
        print("Loading vectors from cache...")
        vectors = np.array(json.loads(CACHE_FILE.read_text()))
        agent.build_bm25(contents)
        return vectors
    
    vectors = agent.embed(contents)
    CACHE_FILE.write_text(json.dumps(vectors))
    print("Vectors cached.")
    return vectors

# RAGAgent class that handles embedding, retrieval, reranking, and answering questions using Gemini and local embeddings.
class RAGAgent:
    """Retrieval-Augmented Generation agent backed by Gemini + local embeddings."""

    #runs automatically when object is created
    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set!")

        self._client = genai.Client(
            api_key=api_key, http_options={"api_version": "v1"}
        )
        #embeds the documents
        self._embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
        #embedding index for BM25 retrieval, built on the raw text (not vectors)
        self._bm25: BM25Okapi | None = None
        #rerank the documents retrieved by BM25 and vector search using a cross-encoder for better relevance to the query
        self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    #embeds the input texts and returns a list of float vectors. Also builds the BM25 index for the same texts.
    def embed(self, texts: list[str] | str) -> list[list[float]]:
        """Encode *texts* and return a list of float vectors."""
        if isinstance(texts, str):
            texts = [texts]
            
        tokenized = [text.lower().split() for text in texts]
        self._bm25 = BM25Okapi(tokenized)
        return [v.tolist() for v in self._embed_model.encode(texts)]

    def retrieve(self, query, documents, vectors, top_k=5):
        # --- Vector scores ---
        query_vector = np.array(self._embed_model.encode([query])[0])
        cosine_scores = _cosine_similarity(np.array(vectors), query_vector)

        # --- BM25 scores (only if index exists) ---
        if self._bm25 is not None:
            bm25_scores = np.array(self._bm25.get_scores(query.lower().split()))

            def normalize(s):
                r = s.max() - s.min()
                return (s - s.min()) / (r if r > 0 else 1)

            fused = 0.5 * normalize(cosine_scores) + 0.5 * normalize(bm25_scores)
        else:
            fused = cosine_scores  # Fallback to pure vector search

        #return the top_k documents based on the fused scores of vector similarity and BM25 relevance
        top_indices = np.argsort(fused)[::-1][:top_k]
        return [documents[i] for i in top_indices]
    
    #rerank the documents and return only the top_k most relevant ones based on the query
    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[str]:
        pairs = [(query, doc) for doc in documents]
        scores = self._reranker.predict(pairs)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [documents[i] for i in top_indices]
    
    #BM25 helps with retrieval by giving higher scores to documents that contain more of the query terms, paired with semantic search
    def build_bm25(self, texts: list[str]) -> None:
        tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        
    def rewrite_query(self, question: str) -> str:
    #Use Gemini to rewrite the user query for better retrieval.
        prompt = _QUERY_REWRITE_TEMPLATE.format(question=question)
        result = self._client.models.generate_content(
            model=_GEMINI_MODEL, contents=prompt
        )
        rewritten = result.text.strip()
        return rewritten

    def answer(self, question, documents, vectors, *, verbose=False):
        #rewrite the user query for better retrieval
        rewritten = self.rewrite_query(question)
        
        if verbose:
            print(f"\n--- Query Rewrite ---")
            print(f"Original:  {question}")
            print(f"Rewritten: {rewritten}")
            print(f"---------------------\n")

        #get documents with query, and then rerank them for final answer
        candidates = self.retrieve(rewritten, documents, vectors, top_k=10)
        relevant = self.rerank(rewritten, candidates, top_k=5)

        if verbose:
            self._print_context(relevant)

        prompt = _RAG_PROMPT_TEMPLATE.format(
            context="\n\n".join(relevant),
            question=question,   # Use original question for the final answer so it makes more sense to the user, but rewritten query for retrieval to get better results from the database
        )
        result = self._client.models.generate_content(
            model=_GEMINI_MODEL, contents=prompt
        )
        return result.text, relevant

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_context(docs: list[str]) -> None:
        print("\n--- Retrieved Context ---")
        for doc in docs:
            print(doc)
        print("------------------------\n")