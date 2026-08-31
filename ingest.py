"""Load lecture documents, embed their chunks, and persist a FAISS index."""

from dataclasses import dataclass

import faiss
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    SUPPORTED_EXTENSIONS,
    VECTOR_STORE_DIR,
)


@dataclass(frozen=True)
class IngestionResult:
    """Summary of a completed ingestion run.

    Attributes:
        source_files: Number of supported files found in the documents folder.
        documents: Number of document sections returned by the file readers.
        embedding_dimension: Number of values in each embedding vector.

    """

    source_files: int
    documents: int
    embedding_dimension: int


def ingest_documents(
    embed_model: BaseEmbedding | None = None,
) -> IngestionResult:
    """Build and persist a FAISS index from the configured document folder.

    Args:
        embed_model: Optional loaded embedding model to reuse. When omitted,
            the configured Hugging Face BGE model is loaded locally.

    Returns:
        Counts and embedding information from the completed ingestion.

    Raises:
        FileNotFoundError: If the configured documents directory is missing.
        ValueError: If the directory contains no supported documents.

    """
    if embed_model is None:
        print("Loading embedding model...")
        embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL,
            normalize=True,
        )
    Settings.embed_model = embed_model

    print("Loading documents...")
    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(f"Could not find '{DOCUMENTS_DIR}' folder.")

    documents = SimpleDirectoryReader(
        input_dir=str(DOCUMENTS_DIR),
        recursive=True,
        required_exts=list(SUPPORTED_EXTENSIONS),
    ).load_data()

    if not documents:
        raise ValueError(f"No supported documents found in '{DOCUMENTS_DIR}'.")

    source_files = sum(
        path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        for path in DOCUMENTS_DIR.rglob("*")
    )
    print(f"Loaded {len(documents)} document(s).")
    print("Creating FAISS index...")

    # Derive the dimension so changing embedding models cannot mismatch FAISS.
    embedding_dimension = len(
        embed_model.get_text_embedding("embedding dimension probe")
    )
    # Normalized vectors make inner product equivalent to cosine similarity.
    faiss_index = faiss.IndexFlatIP(embedding_dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Chunking documents and creating embeddings...")
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    # LlamaIndex splits each document, embeds every node, and writes it to FAISS.
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
    )

    print("Saving vector store...")
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(VECTOR_STORE_DIR))

    return IngestionResult(
        source_files=source_files,
        documents=len(documents),
        embedding_dimension=embedding_dimension,
    )


def main() -> None:
    """Run ingestion from the command line and print its summary."""
    result = ingest_documents()

    print()
    print("========================================")
    print("Ingestion complete!")
    print("========================================")
    print(f"Source files:  {result.source_files}")
    print(f"Documents:     {result.documents}")
    print(f"Embedding:     {EMBEDDING_MODEL}")
    print(f"Dimensions:    {result.embedding_dimension}")
    print(f"Chunk size:    {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")
    print(f"Vector store:  {VECTOR_STORE_DIR}")
    print("========================================")


if __name__ == "__main__":
    main()
