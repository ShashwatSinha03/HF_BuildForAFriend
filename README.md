# StudyLens

A lightweight, local-first AI study assistant that runs entirely on your machine using Streamlit + Ollama.

## What it does

StudyLens helps you turn study material into active learning:

- **Understand** — Ask questions about your material, get grounded answers
- **Revise** — Generate a concise, exam-friendly revision sheet
- **Quiz** — Test yourself with MCQs and short-answer questions, get scored
- **Exam Mode** — Get a prioritized study plan for the time you have (15/30/60 min)

All processing happens locally. No cloud, no API keys, no accounts.

## Requirements

- **Python 3.10+**
- **Ollama** with **Gemma 3 1B** model

### Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

### Pull the model

```bash
ollama pull gemma3:1b
```

### Start Ollama server

```bash
ollama serve
```

## Setup

```bash
# Clone the repository
git clone <repository-url>
cd study-lens

# Install Python dependencies
pip install -r requirements.txt
```

## Running

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

## Usage

1. **Upload** a PDF or TXT file
2. **Understand** — Ask specific questions, get grounded answers
3. **Revise** — Generate a structured revision sheet
3. **Quiz** — Choose 5 or 10 questions, answer, get scored with weak areas
4. **Exam Mode** — Select time (15/30/60 min), get a prioritized study plan

## Project Structure

```
study-lens/
├── app.py          # Main Streamlit application
├── llm.py          # Ollama connection & generation
├── document.py     # Document extraction & retrieval
├── requirements.txt
├── README.md
└── .gitignore
```

## Features

- ✅ PDF/TXT upload with text extraction
- ✅ Lightweight keyword-based retrieval (no embeddings/vector DB)
- ✅ Understand — grounded Q&A with citations
- ✅ Revise — structured revision sheets (concepts, terms, relationships, remember, confusions)
- ✅ Quiz — MCQs + short answers, batch evaluation, weak areas
- ✅ Exam Mode — 15/30/60 min prioritized study plans (Must Know / Good to Know / Skim)
- ✅ Local-first: Ollama + Gemma 3 1B, no cloud
- ✅ Session state resets on new document upload

## Limitations

- Gemma 3 1B is a small local model — answers depend on document quality
- Retrieval is keyword-based (not semantic) — may miss related concepts
- Scanned/image-only PDFs are not supported (no OCR)
- Structured generation can occasionally fail — retry button available
- No persistent storage — everything resets on browser close

## License

MIT