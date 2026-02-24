import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from pathlib import Path

#get the path to the service account key file
ROOT_DIR = Path(__file__).parent

#initialize the Firebase app with the service account key
cert = ROOT_DIR / "utsa-gpt-firebase-adminsdk-fbsvc-d2f7d92bc8.json"
cred = credentials.Certificate(cert)
firebase_admin.initialize_app(cred)

#get a reference to the Firestore client
db = firestore.client()

def get_document(name):
    try:
        #query the test_collection for a document
        docs = db.collection(name).limit(1).stream()
        
        found = False
        #go through the documents returned by the query, currently limited by the .limit(1)
        for doc in docs:
            data = doc.to_dict()
            print(f"Connected to Firebase. Found document: {doc.id}")
            
            found = True
            #can change to if doc to remove found variable and just check if doc exists, but this is more explicit for now
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
            'day': row["Day"],
            "date": row["Date"],
            "away": row["Away"],
            "home": row["Home"]
        }
        
        db.collection("game").document(doc_id).set(game_data)
    print("Data write complete!")