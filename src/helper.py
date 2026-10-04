from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------------------------------------
# Extract data from PDF files
# ---------------------------------------------------------
def load_pdf_file(data: str | Path) -> List[Document]:
    """
    Load all PDF files from the specified directory.
    """

    data = Path(data)

    if not data.exists():
        raise FileNotFoundError(
            f"PDF directory not found: {data.resolve()}"
        )

    loader = DirectoryLoader(
        str(data),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        silent_errors=False
    )

    documents = loader.load()

    return documents


# ---------------------------------------------------------
# Keep only required metadata
# ---------------------------------------------------------
def filter_to_minimal_docs(
    docs: List[Document]
) -> List[Document]:
    """
    Keep the original page content and only the source metadata.
    """

    minimal_docs = []

    for doc in docs:
        src = doc.metadata.get("source")

        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": src
                }
            )
        )

    return minimal_docs


# ---------------------------------------------------------
# Split documents into chunks
# ---------------------------------------------------------
def text_split(
    extracted_data: List[Document]
) -> List[Document]:
    """
    Split documents into smaller overlapping chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )

    text_chunks = text_splitter.split_documents(
        extracted_data
    )

    return text_chunks


# ---------------------------------------------------------
# Download / initialize Hugging Face embeddings
# ---------------------------------------------------------
def download_hugging_face_embeddings():
    """
    Initialize the Hugging Face embedding model.

    all-MiniLM-L6-v2 produces 384-dimensional embeddings.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings