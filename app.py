import gradio as gr

from rag import ask

# ============================================================
# Chat function
# ============================================================


def chat(message, history):
    """
    Send the user's question to the RAG pipeline.

    history is provided by Gradio so the UI can maintain
    conversation history visually.
    """

    if not message.strip():
        return ""

    try:
        answer = ask(message)
        return answer

    except Exception as e:
        return f"❌ Error: {e}"


# ============================================================
# Gradio UI
# ============================================================

demo = gr.ChatInterface(
    fn=chat,
    title="📚 Local RAG Chatbot",
    description=(
        "Ask questions about your local PDF, DOCX, PPTX, TXT, and Markdown documents."
    ),
    textbox=gr.Textbox(
        placeholder="Ask something about your documents...",
        container=True,
    ),
)


# ============================================================
# Start application
# ============================================================

if __name__ == "__main__":
    demo.launch()
