import asyncio
import os
import warnings
import logging
import json
import hashlib
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

#file with embedded vectors to avoid re-embedding
CACHE_FILE = Path(__file__).parent / "vectors_cache.json"

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
You are given the original question and a rewritten query that will provide clarity,
but answer the question based off the original question.
Do NOT use any outside knowledge. If the answer is not in the data, say \
"I don't have that information."

Here is the data:
{context}

Question: {question}

Answer using ONLY the data above:"""

# Generic query rewrite template used when no custom template file is provided.
# To tailor retrieval for a specific database, pass a .txt file path to RAGAgent
# that overrides this template. The file must contain {question} as a placeholder.
_DEFAULT_QUERY_REWRITE_TEMPLATE = """\
You are an expert at reformulating questions to improve document retrieval.
Given a user's question, rewrite it to be more specific and keyword-rich \
for searching the database.

Rules:
- Make implicit information explicit
- Keep it concise — one sentence
- Return ONLY the rewritten query, no explanation

Original question: {question}
Rewritten query:"""


# performs cosine similarity between the query vector and each document vector to get relevance scores for retrieval
def _cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Return cosine similarities between every row of *matrix* and *vector*."""
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    return np.dot(matrix, vector) / np.where(norms == 0, 1e-10, norms)


# Compute a hash of the document contents to detect changes for cache invalidation.
def _hash_contents(contents: list[str]) -> str:
    combined = "\n".join(contents).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()


# Load vectors from cache if available and documents haven't changed, otherwise embed and cache them.
def load_or_embed(agent, contents):
    current_hash = _hash_contents(contents)

    if CACHE_FILE.exists():
        cached = json.loads(CACHE_FILE.read_text())
        if cached.get("hash") == current_hash:
            print("Loading vectors from cache...")
            agent._build_bm25(contents)
            return np.array(cached["vectors"])
        print("Documents changed — re-embedding...")

    vectors = agent.embed(contents)
    CACHE_FILE.write_text(json.dumps({"hash": current_hash, "vectors": vectors}))
    print("Vectors cached.")
    return np.array(vectors)


# Module-level singleton — initialized once at FastAPI startup via get_agent().
# Loading SentenceTransformer and CrossEncoder takes several seconds; creating a
# new RAGAgent per-request would make every request pay that cost. The singleton
# ensures models are loaded exactly once when the server starts.
_agent: "RAGAgent | None" = None


def get_agent(query_rewrite_template: "str | Path | None" = None) -> "RAGAgent":
    """Return the shared RAGAgent singleton, creating it on first call.
    Intended to be called once inside FastAPI's lifespan startup handler so the
    heavy model loading happens at server start, not on the first request.
    """
    global _agent
    if _agent is None:
        _agent = RAGAgent(query_rewrite_template=query_rewrite_template)
    return _agent


# RAGAgent class that handles embedding, retrieval, reranking, and answering questions using Gemini and local embeddings.
class RAGAgent:
    """Retrieval-Augmented Generation agent backed by Gemini + local embeddings.

    Heavy models (SentenceTransformer, CrossEncoder) are loaded once in __init__.
    Use get_agent() to access the module-level singleton rather than instantiating
    this class directly — this ensures models are only loaded once per process.
    """

    def __init__(self, query_rewrite_template: "str | Path | None" = None) -> None:
        """
        Args:
            query_rewrite_template: Optional path to a .txt file containing a
                custom query rewrite prompt for a specific database. The file must
                include {question} as a placeholder. If omitted, the generic default
                template is used. See nfl_query_rewrite.txt for an example.
        """
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

        if query_rewrite_template is not None:
            self._query_rewrite_template = Path(query_rewrite_template).read_text()
        else:
            self._query_rewrite_template = _DEFAULT_QUERY_REWRITE_TEMPLATE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: list[str] | str) -> list[list[float]]:
        """Encode *texts* into dense vectors AND build the BM25 sparse index.

        Both indexes are always built together so they stay in sync.
        Call this once with the full document corpus before calling retrieve().
        """
        if isinstance(texts, str):
            texts = [texts]
        self._build_bm25(texts)
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

    def _build_bm25(self, texts: list[str]) -> None:
        """Build the BM25 sparse index from *texts*.

        Called internally by embed(). Also called directly by load_or_embed()
        when vectors are restored from cache, since BM25 is not cached and must
        be rebuilt from the raw document text.
        """
        tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    async def rewrite_query(self, question: str) -> str:
        """Use Gemini to rewrite the user query for better retrieval.

        Runs the blocking Gemini API call in a thread pool so the FastAPI event
        loop is not blocked while waiting for the network response.
        """
        prompt = self._query_rewrite_template.format(question=question)
        result = await asyncio.to_thread(
            self._client.models.generate_content,
            model=_GEMINI_MODEL,
            contents=prompt,
        )
        return result.text.strip()

    async def answer(self, question, documents, vectors, *, verbose=False):
        """Retrieve relevant documents and generate a grounded answer via Gemini.

        This method is async because it contains multiple blocking operations:
        - rewrite_query: network call to the Gemini API
        - retrieve: runs SentenceTransformer inference (CPU-bound)
        - rerank: runs CrossEncoder inference (CPU-bound)
        - generate_content: another network call to the Gemini API

        All blocking calls are offloaded to a thread pool via asyncio.to_thread()
        so the FastAPI event loop remains free to handle other requests while
        waiting. This means the server stays responsive under concurrent load
        rather than freezing for the duration of each Gemini/model call.
        """
        #rewrite the user query for better retrieval
        rewritten = await self.rewrite_query(question)

        if verbose:
            print(f"\n--- Query Rewrite ---")
            print(f"Original:  {question}")
            print(f"Rewritten: {rewritten}")
            print(f"---------------------\n")

        #get documents with query, and then rerank them for final answer
        candidates = await asyncio.to_thread(self.retrieve, rewritten, documents, vectors, 10)
        relevant = await asyncio.to_thread(self.rerank, rewritten, candidates, 5)

        # Pass both the original and rewritten query to Gemini so it answers the
        # user's intent while benefiting from the clarified retrieval query.
        prompt_question = "Original question: " + question + "\nRewritten query: " + rewritten

        if verbose:
            self._print_context(relevant)

        prompt = _RAG_PROMPT_TEMPLATE.format(
            context="\n\n".join(relevant),
            question=prompt_question,
        )
        result = await asyncio.to_thread(
            self._client.models.generate_content,
            model=_GEMINI_MODEL,
            contents=prompt,
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
