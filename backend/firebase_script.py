import firebase_admin
from firebase_admin import credentials, firestore as admin_firestore
import pandas as pd
from pathlib import Path
from llama_index.core import Document

ROOT_DIR = Path(__file__).parent

cert = ROOT_DIR / "utsa-gpt-firebase-adminsdk-fbsvc-d2f7d92bc8.json"
cred = credentials.Certificate(cert)
firebase_admin.initialize_app(cred)

db = admin_firestore.client()


def get_document(name):
    try:
        docs = db.collection(name).limit(1).stream()
        found = False
        for doc in docs:
            data = doc.to_dict()
            print(f"Connected to Firebase. Found document: {doc.id}")
            found = True
        if found:
            return data
        else:
            print("Connected to Firebase. No document found")
    except Exception as e:
        print("Failed to connect to Firebase:")
        print(e)


def write_to_db(file):
    df = pd.read_csv(file)
    for _, row in df.iterrows():
        doc_id = f"{row['Date']}_{row['Away']}_{row['Home']}"
        game_data = {
            "day": row["Day"],
            "date": row["Date"],
            "away": row["Away"],
            "home": row["Home"],
        }
        db.collection("game").document(doc_id).set(game_data)
    print("Data write complete!")


def llamaLoad(collection):
    docs = db.collection(collection).stream()
    documents = []
    for doc in docs:
        data = doc.to_dict()
        # Format as natural sentence for better embedding similarity
        text = (
            f"On {data.get('day', '')} {data.get('date', '')}, "
            f"{data.get('away', '')} plays at {data.get('home', '')}."
        )
        documents.append(Document(text=text, metadata={"doc_id": doc.id}))
    return documents