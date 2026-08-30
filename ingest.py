from pathlib import Path

import faiss
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

# ============================================================
# Configuration
# ============================================================

DOCUMENTS_DIR = Path("documents")
VECTOR_STORE_DIR = Path("vector_store")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


# ============================================================
# 1. Set up embedding model
# ============================================================

print("Loading embedding model...")

Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)


# ============================================================
# 2. Load documents
# ============================================================

print("Loading documents...")

if not DOCUMENTS_DIR.exists():
    raise FileNotFoundError(f"Could not find '{DOCUMENTS_DIR}' folder.")

documents = SimpleDirectoryReader(
    input_dir=str(DOCUMENTS_DIR),
    recursive=True,
    required_exts=[
        ".pdf",
        ".docx",
        ".pptx",
        ".txt",
        ".md",
    ],
).load_data()

if not documents:
    raise ValueError(f"No supported documents found in '{DOCUMENTS_DIR}'.")

print(f"Loaded {len(documents)} document(s).")


# ============================================================
# 3. Create FAISS index
# ============================================================

print("Creating FAISS index...")

# BGE-small-en-v1.5 produces 384-dimensional embeddings
embedding_dimension = 384

faiss_index = faiss.IndexFlatL2(embedding_dimension)

vector_store = FaissVectorStore(faiss_index=faiss_index)


# ============================================================
# 4. Create storage context
# ============================================================

storage_context = StorageContext.from_defaults(vector_store=vector_store)


# ============================================================
# 5. Chunk documents and build index
# ============================================================

print("Chunking documents and creating embeddings...")

splitter = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    transformations=[splitter],
)


# ============================================================
# 6. Save FAISS index + metadata
# ============================================================

print("Saving vector store...")

VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

index.storage_context.persist(persist_dir=str(VECTOR_STORE_DIR))


# ============================================================
# Done
# ============================================================

print()
print("========================================")
print("✅ Ingestion complete!")
print("========================================")
print(f"Documents:    {len(documents)}")
print(f"Embedding:    {EMBEDDING_MODEL}")
print(f"Chunk size:   {CHUNK_SIZE}")
print(f"Chunk overlap:{CHUNK_OVERLAP}")
print(f"Vector store: {VECTOR_STORE_DIR}")
print("========================================")
