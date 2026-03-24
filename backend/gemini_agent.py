import os
import warnings
import logging

import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

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

load_dotenv()

# Ensure requests go to the public Gemini API, not Vertex AI
for _key in ("GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI"):
    os.environ.pop(_key, None)

_GEMINI_MODEL = "gemini-2.5-flash"
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_RAG_PROMPT_TEMPLATE = """\
You are a helpful assistant that ONLY answers questions using the provided data.
Do NOT use any outside knowledge. You are permitted to use some context clues such as knowing that 
"The Cowboys" is a professional American football team in "Dallas, Texas".
If the answer is not in the data, say \
"I don't have that information."

Here is the data:
{context}

Question: {question}

Answer using ONLY the data above:"""


def _cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Return cosine similarities between every row of *matrix* and *vector*."""
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    return np.dot(matrix, vector) / np.where(norms == 0, 1e-10, norms)


class RAGAgent:
    """Retrieval-Augmented Generation agent backed by Gemini + local embeddings."""

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set!")

        self._client = genai.Client(
            api_key=api_key, http_options={"api_version": "v1"}
        )
        self._embed_model = SentenceTransformer(_EMBED_MODEL_NAME)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: list[str] | str) -> list[list[float]]:
        """Encode *texts* and return a list of float vectors."""
        if isinstance(texts, str):
            texts = [texts]
        return [v.tolist() for v in self._embed_model.encode(texts)]

    def retrieve(
        self,
        query: str,
        documents: list[str],
        vectors: list[list[float]],
        top_k: int = 3,
    ) -> list[str]:
        """Return the *top_k* most relevant documents for *query*."""
        query_vector = np.array(self._embed_model.encode([query])[0])
        similarities = _cosine_similarity(np.array(vectors), query_vector)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [documents[i] for i in top_indices]

    def answer(
        self,
        question: str,
        documents: list[str],
        vectors: list[list[float]],
        *,
        verbose: bool = False,
    ) -> tuple[str, list[str]]:
        """Run a full RAG pipeline and return *(answer, retrieved_docs)*."""
        relevant = self.retrieve(question, documents, vectors)
        if verbose:
            self._print_context(relevant)

        prompt = _RAG_PROMPT_TEMPLATE.format(
            context="\n\n".join(relevant),
            question=question,
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