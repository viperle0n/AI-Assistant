"""This script loads the FAISS vector database and performs semantic search over the insurance policies"""

import re
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Location of the vector database
VECTORSTORE_PATH = "vectorstore"

# Same embedding model used during ingestion
EMBEDDING_MODEL = "nomic-embed-text"

def load_vectorstore():

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore

"""
Why allow dangerous deserialization?
FAISS stores metadata inside a pickle file. Python warns that loading arbitrary pickle files can execute code.
In our own trusted application it's acceptable because we created the index ourselves.
If this were production software loading indexes from unknown sources, we'd use a safer serialization mechanism.
"""


def extract_requested_year(question: str):
    """Detects whether the user explicitly requested a policy year"""

    match = re.search(r"(20\d{2})", question)

    if match:
        return match.group(1)

    return None

def retrieve_policy_evidence(
    question: str,
    requested_year=None,
    k: int = 5
):
    
    """
    Retrieves the most relevant policy chunks based on the user's question. (k->chunks)     
    Returns a list of relevant documents with metadata.
    """

    # Load FAISS database
    vectorstore = load_vectorstore()

    # Check if the user requested a specific year
    # Prefer the conversation context
    # If no context exists, check the question
    if requested_year is None:
        requested_year = extract_requested_year(question)


    print(f"User requested year: {requested_year}")

    if requested_year:

        # Search only inside the requested policy year
        results = vectorstore.similarity_search_with_score(
            question,
            k=k,
            filter={
                "year": requested_year
            }
        )

    else:

        # No year specified.
        # Retrieve from all documents.
        results = vectorstore.similarity_search_with_score(
            question,
            k=k
        )

    evidence = []


    for doc, score in results:
        
        print("\n========================")
        print("Similarity score:", score)
        print("Source:", doc.metadata.get("source"))
        print("Year:", doc.metadata.get("year"))
        print("Page:", doc.metadata.get("page"))

        print("\nContent:")
        print(doc.page_content)
        evidence.append(
            {
                "content": doc.page_content,

                "metadata": doc.metadata,

                "score": float(score)
            }
    )


    return evidence

        
