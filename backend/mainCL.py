import traceback
from firebase_script import load_documents
from gemini_agent import RAGAgent, load_or_embed

# collection is the name of the database, inside each database is a 'document' which is a dictionary with keys and values.
_COLLECTION = "game"
#command to exit from the cli
_EXIT_COMMAND = "exit"

_GREETING = "Ask a question about the NFL schedule, and I'll do my best to answer it based on the data I have!"


def mainCL() -> None:
    try:
        # load documents as a list of LlamaIndex Document objects, which have the string and name of document it is from
        documents = load_documents(_COLLECTION)
        if not documents:
            print("No documents found in the collection.")
            return
        print(f"Documents loaded: {len(documents)}")

        # object of RAGAgent class, which has methods to embed, retrieve, rerank, and answer questions based on the documents
        agent = RAGAgent()

        #turn the documents into a list of strings, then finds vector cache, or embeds if not done yet
        contents = [doc.text for doc in documents]
        vectors = load_or_embed(agent, contents)
        print(f"Indexed {len(vectors)} documents.")

        #while loop for cli to ask questions continually until 'exit'
        while True:
            question = input(
                _GREETING + 
                f"(or type '{_EXIT_COMMAND}' to quit): "
            ).strip()

            if not question:
                continue
            if question.lower() == _EXIT_COMMAND:
                break

            #start the retrieval and answer generation process inside gemini_agent.py
            answer, _ = agent.answer(question, contents, vectors, verbose=True)
            print("\nGemini Agent Response:", answer)

    except Exception:
        traceback.print_exc()

#ensure that main() runs when this file is executed directly, but not when imported as a module (e.g., for testing)
if __name__ == "__main__":
    main()