"""Gradio interface for chatting with and re-indexing lecture documents."""

import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import gradio as gr

from config import DOCUMENTS_DIR, SUPPORTED_EXTENSIONS
from ingest import ingest_documents
from rag import embed_model, index_is_ready, reload_index, stream_answer


def chat(
    message: str,
    history: Sequence[Any] | None,
) -> Iterator[str]:
    """Stream a RAG response for a Gradio chat message.

    Args:
        message: The user's latest question.
        history: Previous messages from the current Gradio session.

    Yields:
        Progressively longer versions of the generated answer.

    """
    if not message.strip():
        return

    yield from stream_answer(message, history)


def _save_uploads(files: list[str] | None) -> list[str]:
    """Copy uploaded lecture files into the documents directory.

    Files with the same name replace the existing copy so users can upload a
    revised version of a lecture.

    Args:
        files: Temporary file paths supplied by Gradio.

    Returns:
        Names of the files saved in the documents directory.

    Raises:
        gr.Error: If no files are selected or a file type is unsupported.

    """
    if not files:
        raise gr.Error("Select at least one lecture document.")

    uploaded_paths = [Path(file) for file in files]
    unsupported = [
        path.name
        for path in uploaded_paths
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS
    ]
    if unsupported:
        names = ", ".join(unsupported)
        raise gr.Error(f"Unsupported file type: {names}")

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for source in uploaded_paths:
        destination = DOCUMENTS_DIR / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        saved_files.append(destination.name)

    return saved_files


def _reindex() -> str:
    """Rebuild the persisted index and refresh the active query engines.

    Returns:
        A Markdown status message describing the completed ingestion.

    Raises:
        FileNotFoundError: If the documents directory does not exist.
        ValueError: If no supported documents are available.

    """
    # Reuse the model loaded by rag.py instead of loading BGE a second time.
    result = ingest_documents(embed_model=embed_model)
    reload_index()
    return (
        f"✅ Indexed **{result.source_files} file(s)** "
        f"into **{result.documents} document section(s)**."
    )


def upload_and_reindex(files: list[str] | None) -> tuple[str, None]:
    """Save uploaded files and rebuild the complete document index.

    Args:
        files: Temporary file paths supplied by Gradio.

    Returns:
        A status message and ``None`` to clear the upload component.

    Raises:
        gr.Error: If the upload selection is empty or unsupported.
        FileNotFoundError: If an uploaded file cannot be accessed.
        ValueError: If ingestion finds no supported documents.

    """
    saved_files = _save_uploads(files)
    status = _reindex()
    names = ", ".join(saved_files)
    return f"Uploaded: **{names}**\n\n{status}", None


def reindex_existing() -> str:
    """Rebuild the index from files already in the documents directory.

    Returns:
        A Markdown status message describing the completed ingestion.

    Raises:
        FileNotFoundError: If the documents directory does not exist.
        ValueError: If no supported documents are available.

    """
    return _reindex()


# Report whether rag.py successfully loaded a persisted index at startup.
initial_status = (
    "✅ FAISS index is ready."
    if index_is_ready()
    else "⚠️ No FAISS index yet. Upload documents or run `python ingest.py`."
)

# Blocks provides a two-column layout around the high-level chat interface.
with gr.Blocks() as demo:
    gr.Markdown("# 📚 Local RAG Chatbot")
    gr.Markdown(
        "Ask questions about your local PDF, DOCX, PPTX, TXT, and Markdown "
        "lecture documents."
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot()
            gr.ChatInterface(
                fn=chat,
                chatbot=chatbot,
                textbox=gr.Textbox(
                    placeholder="Ask something about your documents...",
                    container=True,
                ),
                concurrency_limit=1,
            )

        with gr.Column(scale=1):
            gr.Markdown("### Documents")
            uploads = gr.File(
                label="Upload lecture documents",
                file_count="multiple",
                file_types=list(SUPPORTED_EXTENSIONS),
                type="filepath",
            )
            upload_button = gr.Button("Upload and re-index", variant="primary")
            reindex_button = gr.Button("Re-index existing documents")
            status = gr.Markdown(initial_status)

    upload_button.click(
        fn=upload_and_reindex,
        inputs=uploads,
        outputs=[status, uploads],
        concurrency_limit=1,
    )
    reindex_button.click(
        fn=reindex_existing,
        outputs=status,
        concurrency_limit=1,
    )

# Serialize expensive local-model and re-indexing work.
demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch()
