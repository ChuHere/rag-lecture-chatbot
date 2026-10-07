"""Load the lecture index and answer questions with local retrieval and Qwen."""

import re
from collections.abc import Iterator, Sequence
from pathlib import Path
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
    EMBEDDING_MODEL,
    LLM_MODEL,
    MAX_CHAT_MESSAGES,
    OLLAMA_URL,
    SIMILARITY_CUTOFF,
    TOP_K,
    VECTOR_STORE_DIR,
)
from prompts import (
    CITATION_QA_TEMPLATE,
    CITATION_REFINE_TEMPLATE,
    NO_RELEVANT_CONTEXT_MESSAGE,
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
    temperature=0.0,
)
Settings.llm = llm

# Replaced as a whole by reload_index(), so readers always see a complete engine.
query_engine: CitationQueryEngine | None = None

EMPTY_RESPONSE = "Empty Response"


def _create_query_engine(index: Any) -> CitationQueryEngine:
    """Create a relevance-filtered, streaming citation query engine.

    Args:
        index: Loaded LlamaIndex vector index.

    Returns:
        A query engine configured for retrieval, filtering, and citations.

    """
    return CitationQueryEngine.from_args(
        index,
        llm=llm,
        similarity_top_k=TOP_K,
        citation_qa_template=CITATION_QA_TEMPLATE,
        citation_refine_template=CITATION_REFINE_TEMPLATE,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)
        ],
        streaming=True,
    )


def reload_index() -> None:
    """Load the persisted FAISS index and replace the active query engine.

    If no persisted index exists, the engine is cleared so the Gradio
    application can still start and offer document uploads.

    Raises:
        ValueError: If persisted LlamaIndex data cannot reconstruct an index.

    """
    global query_engine

    if not VECTOR_STORE_DIR.exists():
        query_engine = None
        return

    print("Loading FAISS index...")
    vector_store = FaissVectorStore.from_persist_dir(str(VECTOR_STORE_DIR))
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=str(VECTOR_STORE_DIR),
    )
    query_engine = _create_query_engine(load_index_from_storage(storage_context))


def index_is_ready() -> bool:
    """Return whether a query engine is currently loaded."""
    return query_engine is not None


def _get_query_engine() -> CitationQueryEngine:
    """Return the active query engine.

    Returns:
        The currently loaded citation query engine.

    Raises:
        FileNotFoundError: If documents have not been indexed yet.

    """
    engine = query_engine
    if engine is None:
        raise FileNotFoundError(
            f"No index found in '{VECTOR_STORE_DIR}'. Upload documents and "
            "re-index them, or run ingest.py first."
        )

    return engine


reload_index()


def _to_chat_messages(history: Sequence[Any] | None) -> list[ChatMessage]:
    """Convert recent Gradio history into LlamaIndex chat messages.

    Args:
        history: Previous ``{"role", "content"}`` messages from Gradio.

    Returns:
        At most ``MAX_CHAT_MESSAGES`` converted user and assistant messages.

    """
    # Bound the history to control prompt size and local-model latency.
    return [
        ChatMessage(role=item["role"], content=item["content"])
        for item in (history or [])[-MAX_CHAT_MESSAGES:]
        if item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
    ]


def _format_sources(source_nodes: Sequence[Any], answer: str) -> str:
    """Format the sources cited in an answer as a Markdown list.

    Retrieval always returns up to ``TOP_K`` chunks, including weak matches the
    answer never uses, so only sources referenced as ``[n]`` are listed. Their
    original numbers are kept so they match the citations in the answer.

    Args:
        source_nodes: Nodes used by the citation query engine.
        answer: The generated answer containing citations such as ``[1]``.

    Returns:
        A Markdown source section, or an empty string when nothing is cited.

    """
    cited = {int(number) for number in re.findall(r"\[(\d+)\]", answer)}
    sources = []

    for number, source_node in enumerate(source_nodes, start=1):
        if number not in cited:
            continue

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
    """Return whether no grounded answer was produced.

    This covers LlamaIndex's placeholder when retrieval finds no usable nodes,
    and the refusal sentence the grounding prompt asks the model to use.
    """
    text = answer.strip()
    return (
        not text
        or text == EMPTY_RESPONSE
        or "couldn't find relevant information" in text.lower()
    )


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
    # A per-request engine prevents conversation memory leaking between users.
    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=_get_query_engine(),
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

    sources = _format_sources(response.source_nodes, answer)
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

        # The last update is the complete answer, including its sources.
        answer = ""
        for answer in stream_answer(question):
            pass

        print()
        print("Qwen:")
        print(answer)
        print()


if __name__ == "__main__":
    main()
