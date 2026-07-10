"""This script reads all PDF files, splits them into smaller chunks, generates embeddings, and stores them inside a vector database"""

import re
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


# Path for PDFs
DOCUMENTS_PATH = Path("app/data/PDF")

# Where FAISS index will be stored
VECTORSTORE_PATH = "vectorstore"

# Embedding model running locally through Ollama
EMBEDDING_MODEL = "nomic-embed-text"


def extract_year(filename: str) -> str:
    """
    Extracts the year from the PDF filename.
    Returns the extracted year as a string.
    Returns "unknown" if no year is found.
    """

    match = re.search(r"(20\d{2})", filename)

    if match:
        return match.group(1)

    return "unknown"


def ingest_documents():
    """Reads every PDF inside the documents folder, splits it into chunks, creates embeddings, and stores everything"""

    
    print("=" * 50)
    print("Current working directory:", os.getcwd())
    print("DOCUMENTS_PATH:", DOCUMENTS_PATH)
    print("Resolved path:", DOCUMENTS_PATH.resolve())
    print("Does folder exist?", DOCUMENTS_PATH.exists())
    print("=" * 50)
    
    # Contains every chunk from every PDF
    documents = []

    # Text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    pdf_files = list(DOCUMENTS_PATH.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF(s):")



    for pdf_file in pdf_files:

        print(f"Looking for PDFs in: {DOCUMENTS_PATH.resolve()}")
        print(list(DOCUMENTS_PATH.glob("*.pdf")))
        print(f" - {pdf}")
        
        year = extract_year(pdf_file.name)

        print(f"Processing: {pdf_file.name}")

        # Load every page from the PDF
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()

        print(f"Loaded {len(pages)} pages.")

        chunks = splitter.split_documents(pages)

        print(f"Created {len(chunks)} chunks.")

        # Attach additional metadata to every chunk
        for index, chunk in enumerate(chunks):

            chunk.metadata.update({
                "year": year,
                "source": pdf_file.name,
                "page": chunk.metadata.get("page"),
                
                # Unique identifier for auditing
                "chunk_id": f"{year}_{index}"
            })

        # Store the processed chunks
        documents.extend(chunks)


    print(f"\nTotal chunks created: {len(documents)}")
    

    # Create the embedding model
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    print("Generating embeddings and building the FAISS index...")

    # FAISS stores:
    #   - the embedding vectors
    #   - the original text
    #   - all metadata
    # Later, the agent will query this database to retrieve the most relevant policy chunks
    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )

    vectorstore.save_local(VECTORSTORE_PATH)

    print("\nVector store created successfully!")

if __name__ == "__main__":
    ingest_documents()












