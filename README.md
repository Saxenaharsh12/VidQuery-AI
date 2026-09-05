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