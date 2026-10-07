# Local Lecture RAG Chatbot

A fully local AI chatbot for asking questions about lecture documents. The application extracts text from your files, splits it into chunks, creates local embeddings, stores them in FAISS, and uses a Qwen model through Ollama to generate answers based on the retrieved lecture content.

After the models have been downloaded, documents and questions remain on your machine.

## Capabilities

- Supports PDF, DOCX, PPTX, TXT, and Markdown files.
- Filters out retrieved chunks that do not meet the relevance threshold.
- Generates cited answers with filenames and page or slide numbers when
  available.
- Understands follow-up questions using recent conversation history.
- Streams Qwen responses into the Gradio chat interface.
- Uploads and re-indexes lecture files directly from the browser.
- Can also run as a terminal chatbot.

## Technology

- [LlamaIndex](https://www.llamaindex.ai/) for document processing and RAG
- [Hugging Face](https://huggingface.co/) BGE embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Ollama](https://ollama.com/) for running Qwen locally
- [Gradio](https://www.gradio.app/) for the browser interface

## Requirements

- Python 3.11
- Ollama
- Enough disk space and memory to run the selected Qwen model

## Local setup

### 1. Create a Python environment

From the project directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Apple Silicon with Homebrew, the Python executable may be:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv
```

### 2. Install and prepare Ollama

Install Ollama from [ollama.com](https://ollama.com/), then download the
language model:

```bash
ollama pull qwen2.5:7b
```

The Ollama server must be running while the chatbot is in use.

### 3. Start the Gradio application

```bash
source .venv/bin/activate
python app.py
```

Open the local URL printed by Gradio, normally:

```text
http://127.0.0.1:7860
```

Use the **Documents** panel to select lecture files and click **Upload and re-index**. When indexing completes, you can ask questions in the chat.

The first run may take longer because the BGE embedding model is downloaded.

## Index documents from the terminal

You can prepare the index before starting Gradio:

1. Copy supported files into the `documents/` directory.
2. Run ingestion.
3. Start the application.

```bash
source .venv/bin/activate
python ingest.py
python app.py
```

Run `python ingest.py` again whenever files in `documents/` change. The browser interface can perform the same full re-index operation.

## Terminal chatbot

After creating the index, you can chat without Gradio:

```bash
source .venv/bin/activate
python rag.py
```

Enter `exit` or `quit` to stop.
