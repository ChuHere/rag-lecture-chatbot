from pathlib import Path

import faiss
from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.faiss import FaissVectorStore

# ============================================================
# Configuration
# ============================================================

VECTOR_STORE_DIR = Path("vector_store")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Change this to the Qwen model you have installed.
# Check with: ollama list
LLM_MODEL = "qwen:latest"

OLLAMA_URL = "http://localhost:11434"

# Number of chunks to retrieve
TOP_K = 5


# ============================================================
# 1. Set up embedding model
# ============================================================

print("Loading embedding model...")

Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)


# ============================================================
# 2. Set up local Qwen model through Ollama
# ============================================================

print("Connecting to Ollama...")

Settings.llm = Ollama(
    model=LLM_MODEL,
    base_url=OLLAMA_URL,
    request_timeout=120.0,
)


# ============================================================
# 3. Load FAISS vector store
# ============================================================

print("Loading FAISS index...")

if not VECTOR_STORE_DIR.exists():
    raise FileNotFoundError(
        f"Could not find '{VECTOR_STORE_DIR}'. Run ingest.py first."
    )


# BGE-small-en-v1.5 produces 384-dimensional embeddings
embedding_dimension = 384

faiss_index = faiss.IndexFlatL2(embedding_dimension)

vector_store = FaissVectorStore.from_persist_dir(str(VECTOR_STORE_DIR))

storage_context = StorageContext.from_defaults(
    vector_store=vector_store,
    persist_dir=str(VECTOR_STORE_DIR),
)


# ============================================================
# 4. Load the LlamaIndex index
# ============================================================

print("Loading document index...")

index = load_index_from_storage(storage_context)


# ============================================================
# 5. Create query engine
# ============================================================

query_engine = index.as_query_engine(
    similarity_top_k=TOP_K,
)


# ============================================================
# 6. Ask questions
# ============================================================


def ask(question: str) -> str:
    """
    Send a question through the RAG pipeline and return the answer.
    """

    print(f"\nQuestion: {question}")

    response = query_engine.query(question)

    print(f"Response object: {response}")
    print(f"Response type: {type(response)}")

    answer = str(response).strip()

    if not answer:
        return "The model returned an empty response."

    return answer


# ============================================================
# 7. Interactive terminal chatbot
# ============================================================


def main():

    print()
    print("========================================")
    print("       Local RAG Chatbot")
    print("========================================")
    print(f"LLM:       {LLM_MODEL}")
    print(f"Embedding: {EMBEDDING_MODEL}")
    print(f"Top K:     {TOP_K}")
    print("========================================")
    print()
    print("Ask questions about your documents.")
    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            answer = ask(question)

            print()
            print("Qwen:")
            print(answer)
            print()

        except Exception as e:
            print()
            print("❌ Error:")
            print(e)
            print()


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()
