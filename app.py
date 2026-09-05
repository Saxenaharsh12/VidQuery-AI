import os
import re
import gradio as gr

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from sentence_transformers import CrossEncoder

from langchain_groq import ChatGroq


# =========================================================
# CONFIGURATION
# =========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Add it in Hugging Face Space Secrets."
    )


# =========================================================
# LOAD MODELS
# =========================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    show_progress=False
)


print("Loading reranker model...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


print("Initializing Groq LLM...")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    api_key=GROQ_API_KEY
)


print("VidQuery AI models loaded successfully!")


# =========================================================
# EXTRACT VIDEO ID
# =========================================================

def extract_video_id(url_or_id):

    url_or_id = url_or_id.strip()

    # Direct video ID
    if re.fullmatch(r"[\w-]{11}", url_or_id):
        return url_or_id

    patterns = [
        r"(?:v=)([\w-]{11})",
        r"youtu\.be/([\w-]{11})",
        r"youtube\.com/shorts/([\w-]{11})",
        r"youtube\.com/embed/([\w-]{11})"
    ]

    for pattern in patterns:

        match = re.search(pattern, url_or_id)

        if match:
            return match.group(1)

    return None


# =========================================================
# GET YOUTUBE TRANSCRIPT
# =========================================================

def get_transcript(video_id):

    ytt_api = YouTubeTranscriptApi()

    transcript_data = ytt_api.fetch(
        video_id,
        languages=["en", "hi"]
    )

    transcript = " ".join(
        snippet.text
        for snippet in transcript_data
    )

    return transcript


# =========================================================
# CREATE VECTOR DATABASE
# =========================================================

def create_vector_store(transcript, video_id):

    documents = [
        Document(
            page_content=transcript,
            metadata={
                "source": "YouTube",
                "video_id": video_id
            }
        )
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store, len(chunks)


# =========================================================
# RETRIEVAL + RERANKING
# =========================================================

def retrieve_and_rerank(query, vector_store, top_k=4):

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 10}
    )

    retrieved_docs = retriever.invoke(query)

    pairs = [
        (query, doc.page_content)
        for doc in retrieved_docs
    ]

    scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(retrieved_docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_docs = [
        doc
        for doc, score in ranked_results[:top_k]
    ]

    return top_docs


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(question, vector_store):

    ranked_docs = retrieve_and_rerank(
        question,
        vector_store
    )

    context = "\n\n".join(
        doc.page_content
        for doc in ranked_docs
    )

    prompt = f"""
You are VidQuery AI, an intelligent YouTube video assistant.

Answer the user's question ONLY using the context provided below.

If the answer cannot be found in the context, clearly say:

"I couldn't find this information in the video."

Context:
{context}

Question:
{question}

Provide a clear, accurate, and concise answer.
"""

    response = llm.invoke(prompt)

    return response.content


# =========================================================
# PROCESS VIDEO
# =========================================================

def process_video(youtube_url):

    if not youtube_url:

        return (
            None,
            "❌ Please enter a YouTube URL."
        )

    try:

        video_id = extract_video_id(youtube_url)

        if not video_id:

            return (
                None,
                "❌ Invalid YouTube URL."
            )


        transcript = get_transcript(video_id)


        vector_store, chunk_count = create_vector_store(
            transcript,
            video_id
        )


        status = (
            f"✅ Video processed successfully!\n\n"
            f"📺 Video ID: {video_id}\n"
            f"📚 Knowledge chunks created: {chunk_count}\n\n"
            f"You can now ask questions about the video!"
        )


        return vector_store, status


    except Exception as e:

        return (
            None,
            f"❌ Error processing video:\n{str(e)}"
        )


# =========================================================
# CHAT FUNCTION
# =========================================================

def chat(question, vector_store, history):

    # If video has not been processed
    if vector_store is None:
        history = history or []

        history.append({
            "role": "assistant",
            "content": "Please process a YouTube video first."
        })

        return history, ""

    # Empty question
    if not question or not question.strip():
        return history, ""

    history = history or []

    try:
        # Get relevant documents from FAISS
        docs = vector_store.similarity_search(question, k=10)

        # Reranking
        pairs = [
            [question, doc.page_content]
            for doc in docs
        ]

        scores = reranker.predict(pairs)

        ranked_docs = [
            doc
            for _, doc in sorted(
                zip(scores, docs),
                key=lambda x: x[0],
                reverse=True
            )
        ]

        # Select best documents
        top_docs = ranked_docs[:5]

        context = "\n\n".join(
            doc.page_content for doc in top_docs
        )

        # Prompt
        prompt = f"""
You are a helpful YouTube video assistant.

Answer the user's question ONLY using the context from the video transcript.

If the answer is not available in the transcript, say:
"I couldn't find this information in the video."

VIDEO CONTEXT:
{context}

QUESTION:
{question}
"""

        # LLM response
        response = llm.invoke(prompt)

        answer = response.content

        # IMPORTANT: New Gradio message format
        history.append({
            "role": "user",
            "content": question
        })

        history.append({
            "role": "assistant",
            "content": answer
        })

        return history, ""

    except Exception as e:

        history.append({
            "role": "assistant",
            "content": f"Error: {str(e)}"
        })

        return history, ""


# =========================================================
# GRADIO UI
# =========================================================

with gr.Blocks(
    title="VidQuery AI"
) as demo:

    gr.Markdown(
        """
        # 🎬 VidQuery AI

        ### Ask intelligent questions about any YouTube video

        **Powered by RAG • FAISS • HuggingFace Embeddings • Cross-Encoder Reranking • Groq LLM**
        """
    )


    vector_store_state = gr.State(None)


    with gr.Row():

        youtube_url = gr.Textbox(
            label="📺 YouTube Video URL",
            placeholder="Paste a YouTube video URL here..."
        )

        process_button = gr.Button(
            "🚀 Process Video",
            variant="primary"
        )


    status = gr.Textbox(
        label="Status",
        interactive=False,
        lines=5
    )


    chatbot = gr.Chatbot(
    label="Video Assistant"
    )



    question = gr.Textbox(
        label="Your Question",
        placeholder="Ask anything about the video..."
    )


    ask_button = gr.Button(
        "Ask Question",
        variant="primary"
    )


    process_button.click(
        fn=process_video,
        inputs=youtube_url,
        outputs=[
            vector_store_state,
            status
        ]
    )


    ask_button.click(
        fn=chat,
        inputs=[
            question,
            vector_store_state,
            chatbot
        ],
        outputs=[
            chatbot,
            question
        ]
    )


    question.submit(
        fn=chat,
        inputs=[
            question,
            vector_store_state,
            chatbot
        ],
        outputs=[
            chatbot,
            question
        ]
    )


    gr.Markdown(
        """
        ---
        Built with ❤️ using **LangChain, FAISS, Hugging Face, Groq and Gradio**
        """
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    import os

    demo.launch(
       server_name="0.0.0.0",
       server_port=int(os.environ.get("PORT", 10000))
       
    )