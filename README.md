# 🎬 VidQuery AI

> An AI-powered YouTube Video Question-Answering application built using **RAG (Retrieval-Augmented Generation)**.

VidQuery AI allows users to paste a YouTube video URL and ask intelligent questions about its content. The application extracts the video's transcript, converts it into searchable embeddings, retrieves the most relevant information, reranks the results, and generates context-aware answers using an LLM.

## 🚀 Live Demo

👉 **Try the application here:**  
https://vidquery-ai-ih7h3xekstna9weox7ybyv.streamlit.app/

---

## ✨ Key Features

- 📺 Process YouTube videos using a video URL
- 📝 Extract video transcripts automatically
- ✂️ Intelligent text chunking for efficient retrieval
- 🧠 Semantic search using Hugging Face embeddings
- 📚 FAISS vector database for similarity search
- 🔍 Two-stage retrieval pipeline
- 🎯 Cross-Encoder reranking for improved answer relevance
- 🤖 Context-aware responses using Groq LLM
- 💬 Interactive conversational interface
- ⚡ Deployed using Streamlit

---

## 🏗️ System Architecture

```text
YouTube Video URL
        │
        ▼
Transcript Extraction
        │
        ▼
Text Chunking
        │
        ▼
Hugging Face Embeddings
        │
        ▼
FAISS Vector Database
        │
        ▼
Semantic Retrieval (Top Results)
        │
        ▼
Cross-Encoder Reranking
        │
        ▼
Relevant Context
        │
        ▼
Groq LLM
        │
        ▼
Final Answer
