"""Load the lecture index and answer questions with local retrieval and Qwen."""

from collections.abc import Iterator, Sequence
from pathlib import Path
from threading import RLock
from typing import Any

from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.faiss import FaissVectorStore

from config import (
    CITATION_CHUNK_SIZE,
    EMBEDDING_MODEL,
    LLM_MODEL,
    MAX_CHAT_MESSAGES,
    OLLAMA_URL,
    SIMILARITY_CUTOFF,
    TOP_K,
    VECTOR_STORE_DIR,
)

print("Loading embedding model...")
embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL,
    normalize=True,
)
Settings.embed_model = embed_model

print("Connecting to Ollama...")
llm = Ollama(
    model=LLM_MODEL,
    base_url=OLLAMA_URL,
    request_timeout=120.0,
    temperature=0.1,
)
Settings.llm = llm

_index_lock = RLock()
query_engine: CitationQueryEngine | None = None
streaming_query_engine: CitationQueryEngine | None = None

EMPTY_RESPONSE = "Empty Response"
NO_RELEVANT_CONTEXT_MESSAGE = (
    "I couldn't find relevant information in the indexed lecture documents."
)


def _create_query_engine(
    index: Any,
    *,
    streaming: bool,
) -> CitationQueryEngine:
    """Create a relevance-filtered citation query engine.

    Args:
        index: Loaded LlamaIndex vector index.
        streaming: Whether Qwen should return generated text incrementally.

    Returns:
        A query engine configured for retrieval, filtering, and citations.

    """
    return CitationQueryEngine.from_args(
        index,
        llm=llm,
        similarity_top_k=TOP_K,
        citation_chunk_size=CITATION_CHUNK_SIZE,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)
        ],
        streaming=streaming,
    )


def reload_index() -> None:
    """Load the persisted FAISS index and replace the active query engines.

    If no persisted index exists, both engine references are cleared so the
    Gradio application can still start and offer document uploads.

    Raises:
        ValueError: If persisted LlamaIndex data cannot reconstruct an index.

    """
    global query_engine, streaming_query_engine

    if not VECTOR_STORE_DIR.exists():
        with _index_lock:
            query_engine = None
            streaming_query_engine = None
        return

    print("Loading FAISS index...")
    vector_store = FaissVectorStore.from_persist_dir(str(VECTOR_STORE_DIR))
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=str(VECTOR_STORE_DIR),
    )
    index = load_index_from_storage(storage_context)
    new_query_engine = _create_query_engine(index, streaming=False)
    new_streaming_query_engine = _create_query_engine(index, streaming=True)

    # Swap both engines together so requests never observe a partial reload.
    with _index_lock:
        query_engine = new_query_engine
        streaming_query_engine = new_streaming_query_engine


def index_is_ready() -> bool:
    """Return whether a non-streaming query engine is currently loaded."""
    with _index_lock:
        return query_engine is not None


def _get_query_engine(*, streaming: bool) -> CitationQueryEngine:
    """Return the requested active query engine.

    Args:
        streaming: Select the streaming engine when true.

    Returns:
        The currently loaded citation query engine.

    Raises:
        FileNotFoundError: If documents have not been indexed yet.

    """
    with _index_lock:
        engine = streaming_query_engine if streaming else query_engine

    if engine is None:
        raise FileNotFoundError(
            f"No index found in '{VECTOR_STORE_DIR}'. Upload documents and "
            "re-index them, or run ingest.py first."
        )

    return engine


reload_index()


def _to_chat_messages(history: Sequence[Any] | None) -> list[ChatMessage]:
    """Convert recent Gradio history into LlamaIndex chat messages.

    Both Gradio's current dictionary format and its older pair format are
    supported.

    Args:
        history: Previous messages from the current Gradio session.

    Returns:
        At most ``MAX_CHAT_MESSAGES`` converted user and assistant messages.

    """
    messages: list[ChatMessage] = []

    # Bound the history to control prompt size and local-model latency.
    for item in (history or [])[-MAX_CHAT_MESSAGES:]:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                messages.append(ChatMessage(role=role, content=content))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_message, assistant_message = item
            if isinstance(user_message, str):
                messages.append(ChatMessage(role="user", content=user_message))
            if isinstance(assistant_message, str):
                messages.append(
                    ChatMessage(role="assistant", content=assistant_message)
                )

    return messages


def _format_sources(source_nodes: Sequence[Any]) -> str:
    """Format retrieved node metadata as a Markdown citation list.

    Args:
        source_nodes: Nodes used by the citation query engine.

    Returns:
        A Markdown source section, or an empty string when no sources exist.

    """
    sources = []

    for number, source_node in enumerate(source_nodes, start=1):
        metadata = source_node.node.metadata
        file_name = metadata.get("file_name")
        if not file_name and metadata.get("file_path"):
            file_name = Path(metadata["file_path"]).name
        file_name = file_name or "Unknown document"

        page = metadata.get("page_label") or metadata.get("page_number")
        location = f", page {page}" if page is not None else ""
        sources.append(f"- [{number}] `{file_name}`{location}")

    if not sources:
        return ""

    return "\n\n**Sources**\n" + "\n".join(sources)


def _is_empty_response(answer: str) -> bool:
    """Return whether LlamaIndex reported that retrieval found no usable nodes."""
    return not answer.strip() or answer.strip() == EMPTY_RESPONSE


def ask(question: str, history: Sequence[Any] | None = None) -> str:
    """Return one complete cited answer for a question.

    Args:
        question: The user's latest question.
        history: Optional conversation history used to resolve follow-ups.

    Returns:
        The complete generated answer followed by its source list.

    Raises:
        FileNotFoundError: If no document index is loaded.

    """
    # A per-request engine prevents conversation memory leaking between users.
    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=_get_query_engine(streaming=False),
        llm=llm,
        chat_history=_to_chat_messages(history),
    )
    response = chat_engine.chat(
        question,
    )
    answer = response.response.strip()

    if _is_empty_response(answer):
        return NO_RELEVANT_CONTEXT_MESSAGE

    return answer + _format_sources(response.source_nodes)


def stream_answer(
    question: str,
    history: Sequence[Any] | None = None,
) -> Iterator[str]:
    """Generate a cited answer and yield updates as Qwen produces text.

    Args:
        question: The user's latest question.
        history: Optional conversation history used to resolve follow-ups.

    Yields:
        Progressively longer answer text, followed by a final sourced answer.

    Raises:
        FileNotFoundError: If no document index is loaded.

    """
    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=_get_query_engine(streaming=True),
        llm=llm,
        chat_history=_to_chat_messages(history),
    )
    response = chat_engine.stream_chat(question)
    answer = ""

    # Gradio expects the full answer-so-far rather than only the newest token.
    for token in response.response_gen:
        answer += token
        if answer.strip() == EMPTY_RESPONSE:
            yield NO_RELEVANT_CONTEXT_MESSAGE
            return
        yield answer

    if _is_empty_response(answer):
        yield NO_RELEVANT_CONTEXT_MESSAGE
        return

    sources = _format_sources(response.source_nodes)
    if sources:
        yield answer + sources


def main() -> None:
    """Run a non-streaming command-line chat loop."""
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

        answer = ask(question)

        print()
        print("Qwen:")
        print(answer)
        print()


if __name__ == "__main__":
    main()
