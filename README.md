<<<<<<< HEAD
---
title: VidQuery AI
emoji: 🎬
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# 🎬 VidQuery AI

An AI-powered YouTube video assistant that allows users to ask intelligent questions about YouTube video content.

## 🚀 Live Demo

👉 [Click here to try VidQuery AI](https://vidquery-ai-ih7h3xekstna9weox7ybyv.streamlit.app/)

## 🚀 Features

- 📺 Process YouTube videos using their URL
- 📝 Extract video transcripts
- ✂️ Intelligent text chunking
- 🧠 Semantic search using Hugging Face embeddings
- 📚 FAISS vector database
- 🔍 Two-stage retrieval pipeline
- 🎯 Cross-Encoder reranking for improved relevance
- 🤖 Context-aware answers using Groq LLM
- 💬 Interactive chat interface using Gradio

---

## 🏗️ Architecture

```text
YouTube URL
     ↓
Transcript Extraction
     ↓
Text Chunking
     ↓
Hugging Face Embeddings
     ↓
FAISS Vector Database
     ↓
Initial Retrieval (Top 10)
     ↓
Cross-Encoder Reranking
     ↓
Top Relevant Context
     ↓
Groq LLM
     ↓
Final Answer
=======
# VidQuery-AI
AI-powered YouTube video chatbot using RAG and Groq.
>>>>>>> 9ed16b815a700e819bea853d8caf2f5f0f69278c
