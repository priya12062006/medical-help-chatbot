from dotenv import load_dotenv
import os
from pathlib import Path

from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings
)

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


# =========================================================
# 1. Load environment variables
# =========================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY is missing. "
        "Please add it to your .env file."
    )


# =========================================================
# 2. Load PDF
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Data folder not found: {DATA_DIR}"
    )

print(f"Loading PDFs from: {DATA_DIR}")

extracted_data = load_pdf_file(DATA_DIR)

print(
    f"Loaded {len(extracted_data)} document pages."
)


# =========================================================
# 3. Clean metadata
# =========================================================

filter_data = filter_to_minimal_docs(
    extracted_data
)


# =========================================================
# 4. Split documents into chunks
# =========================================================

text_chunks = text_split(
    filter_data
)

print(
    f"Created {len(text_chunks)} text chunks."
)


# =========================================================
# 5. Create Hugging Face embeddings
# =========================================================

embeddings = download_hugging_face_embeddings()

print("Hugging Face embedding model loaded.")


# =========================================================
# 6. Connect to Pinecone
# =========================================================

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index_name = "medical-chatbot"

EMBEDDING_DIMENSION = 384


# =========================================================
# 7. Create Pinecone index if it doesn't exist
# =========================================================

if not pc.has_index(index_name):

    print(
        f"Creating Pinecone index: {index_name}"
    )

    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )

    print("Pinecone index created.")

else:

    print(
        f"Pinecone index '{index_name}' already exists."
    )


# =========================================================
# 8. Connect to index
# =========================================================

index = pc.Index(index_name)

print(
    f"Connected to Pinecone index: {index_name}"
)


# =========================================================
# 9. Upload documents
# =========================================================

print("Uploading document chunks to Pinecone...")

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print(
    f"Successfully uploaded {len(text_chunks)} chunks."
)

print("Medical vector database is ready!")