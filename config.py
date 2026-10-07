"""Shared configuration for ingestion, retrieval, and the Gradio interface."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DOCUMENTS_DIR = BASE_DIR / "documents"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".pptx", ".txt", ".md")

EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"


def _resolve_embedding_model() -> str:
    """Prefer a cached local copy of the model before using the remote ID.

    Loading from the cached folder avoids contacting Hugging Face, which some
    networks block even when the model is already downloaded.
    """
    cache_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    model_cache = cache_home / "hub" / "models--BAAI--bge-small-en-v1.5"
    main_ref = model_cache / "refs" / "main"

    if main_ref.is_file():
        revision = main_ref.read_text(encoding="utf-8").strip()
        snapshot = model_cache / "snapshots" / revision
        if snapshot.is_dir():
            return str(snapshot)

    return EMBEDDING_MODEL_ID


EMBEDDING_MODEL = _resolve_embedding_model()
LLM_MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5
SIMILARITY_CUTOFF = 0.45
MAX_CHAT_MESSAGES = 8
