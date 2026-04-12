import os
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore as admin_firestore
from llama_index.core import Document
import pandas as pd

#intialize connection to firebase with key
ROOT_DIR = Path(__file__).parent
_CERT_PATH = Path(os.getenv("FIREBASE_CERT_PATH", ROOT_DIR / "service-account.json"))

#connect to firebase and return a a clinet for the database
def _get_db() -> admin_firestore.Client:
    """Initialize Firebase app once and return a Firestore client."""
    if not firebase_admin._apps:
        cred = credentials.Certificate(_CERT_PATH)
        firebase_admin.initialize_app(cred)
    return admin_firestore.client()

#get a single document from the collection, should be used for testing
def get_document(collection: str) -> dict | None:
    """Return the first document in *collection* as a dict, or None."""
    db = _get_db()
    try:
        for doc in db.collection(collection).limit(1).stream():
            print(f"Connected to Firebase. Found document: {doc.id}")
            return doc.to_dict()
        print("Connected to Firebase. No document found.")
        return None
    except Exception as exc:
        print(f"Failed to connect to Firebase: {exc}")
        return None

#write the nfl schedule data from csv to the database, can be rewritten later for reuseability
def write_to_db(file: str, collection: str = "game") -> None:
    """Upload every row of *file* (CSV) to Firestore under *collection*."""
    db = _get_db()
    df = pd.read_csv(file)
    for _, row in df.iterrows():
        doc_id = f"{row['Date']}_{row['Away']}_{row['Home']}"
        db.collection(collection).document(doc_id).set(
            {
                "day": row["Day"],
                "date": row["Date"],
                "away": row["Away"],
                "home": row["Home"],
            }
        )
    print("Data write complete!")

#load documents from the database and return them as a list of document objects with text and metadata (doc id)
def load_documents(collection: str) -> list[Document]:
    """Stream *collection* from Firestore and return LlamaIndex Documents.

    Each document's fields are serialized generically as "key: value" pairs so
    this function works with any Firestore collection, not just NFL schedule data.
    """
    db = _get_db()
    documents = []
    for doc in db.collection(collection).stream():
        text = ", ".join(f"{k}: {v}" for k, v in doc.to_dict().items())
        documents.append(Document(text=text, metadata={"doc_id": doc.id}))
    return documents