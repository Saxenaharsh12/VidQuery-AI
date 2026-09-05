import os
import re
import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from sentence_transformers import CrossEncoder

from langchain_groq import ChatGroq


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="VidQuery AI",
    page_icon="🎬",
    layout="wide"
)


# =========================================================
# CONFIGURATION
# =========================================================

GROQ_API_KEY = None

# Streamlit Cloud Secrets
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    # Local environment fallback
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")


if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing. Add it to Streamlit Secrets.")
    st.stop()


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

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

    return embeddings, reranker, llm


# Load models
embeddings, reranker, llm = load_models()


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

        return None, "❌ Please enter a YouTube URL."

    try:

        video_id = extract_video_id(youtube_url)

        if not video_id:

            return None, "❌ Invalid YouTube URL."


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

        return None, f"❌ Error processing video:\n{str(e)}"


# =========================================================
# CHAT FUNCTION
# =========================================================

def get_answer(question, vector_store):

    if vector_store is None:
        return "Please process a YouTube video first."

    if not question or not question.strip():
        return "Please enter a question."

    try:

        docs = vector_store.similarity_search(
            question,
            k=10
        )

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

        top_docs = ranked_docs[:5]

        context = "\n\n".join(
            doc.page_content
            for doc in top_docs
        )

        prompt = f"""
You are a helpful YouTube video assistant.

Answer the user's question ONLY using the context from the video transcript.

If the answer is not available in the transcript, say:

"I couldn't find this information in the video."

VIDEO CONTEXT:
{context}

QUESTION:
{question}

Provide a clear, accurate and concise answer.
"""

        response = llm.invoke(prompt)

        return response.content

    except Exception as e:

        return f"❌ Error: {str(e)}"


# =========================================================
# INITIALIZE SESSION STATE
# =========================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# STREAMLIT UI
# =========================================================

st.title("🎬 VidQuery AI")

st.subheader(
    "Ask intelligent questions about any YouTube video"
)

st.caption(
    "Powered by RAG • FAISS • Hugging Face • Cross-Encoder • Groq LLM"
)

st.divider()


# =========================================================
# VIDEO PROCESSING SECTION
# =========================================================

youtube_url = st.text_input(
    "📺 YouTube Video URL",
    placeholder="Paste a YouTube video URL here..."
)


if st.button("🚀 Process Video", use_container_width=True):

    if youtube_url:

        with st.spinner(
            "Processing video and creating knowledge base..."
        ):

            vector_store, status = process_video(
                youtube_url
            )

            if vector_store is not None:

                st.session_state.vector_store = vector_store

                # Clear old chat when new video is processed
                st.session_state.messages = []

                st.success(status)

            else:
                st.error(status)

    else:
        st.warning("Please enter a YouTube URL.")


st.divider()


# =========================================================
# CHAT SECTION
# =========================================================

st.subheader("💬 Ask Questions About the Video")


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# Chat input
question = st.chat_input(
    "Ask anything about the processed video..."
)


if question:

    if st.session_state.vector_store is None:

        st.warning(
            "⚠️ Please process a YouTube video first."
        )

    else:

        # Display user message
        with st.chat_message("user"):

            st.write(question)


        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        # Generate assistant response
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer = get_answer(
                    question,
                    st.session_state.vector_store
                )

                st.write(answer)


        # Save assistant message
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    Built with ❤️ using **LangChain, FAISS, Hugging Face,
    Groq and Streamlit**
    """
)