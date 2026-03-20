import os
import warnings
import numpy as np
from google import genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

# Remove any env vars that would route to Vertex AI
os.environ.pop("GOOGLE_API_KEY", None)
os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set!")

client = genai.Client(api_key=gemini_api_key, http_options={"api_version": "v1"})

_embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def response(content):
    result = client.models.generate_content(
        model="gemini-2.5-flash", contents=content
    )
    return result.text


def embedding(content):
    if isinstance(content, list):
        text = content
    else:
        text = [content]
    vectors = _embed_model.encode(text)
    return [v.tolist() for v in vectors]


def retrieval(query, documents, vectors, top_k=3):
    query_vector = np.array(_embed_model.encode([query])[0])
    doc_vectors = np.array(vectors)
    similarities = np.dot(doc_vectors, query_vector) / (
        np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector)
    )
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [documents[i] for i in top_indices]


def rag_response(question, documents, vectors):
    relevant_docs = retrieval(question, documents, vectors)

    print("\n--- Retrieved Context ---")
    for doc in relevant_docs:
        print(doc)
    print("------------------------\n")

    context = "\n\n".join(relevant_docs)

    prompt = f"""You are a helpful assistant that ONLY answers questions using the provided data.
Do NOT use any outside knowledge. If the answer is not in the data, say "I don't have that information."

Here is the data:
{context}

Question: {question}

Answer using ONLY the data above:"""

    return response(prompt)