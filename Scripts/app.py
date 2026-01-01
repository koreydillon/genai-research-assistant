# app.py

import pickle
import faiss
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# =========================
# Streamlit page + CSS fixes
# =========================
st.set_page_config(page_title="SEC 10-K GenAI Assistant", layout="wide")

st.markdown(
    """
    <style>
    /* Force wrapping for long text (numbers/URLs/table-like strings) to prevent cutoff */
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span {
        word-wrap: break-word !important;
        overflow-wrap: anywhere !important;
        white-space: normal !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# OpenAI client (Streamlit secrets)
# =========================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# =========================
# Load vector store + chunks
# =========================
index = faiss.read_index("vector_store/faiss_index")
with open("vector_store/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# =========================
# Load embedding model
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# UI
# =========================
st.title("📊 SEC 10-K GenAI Assistant")
query = st.text_input("Ask a question about the document:")

# =========================
# Query -> Retrieve -> Answer
# =========================
if query:
    # Embed query (FAISS expects float32)
    query_embedding = model.encode([query]).astype("float32")
    _, I = index.search(query_embedding, k=3)

    # Retrieve top chunks
    retrieved_chunks = [chunks[i] for i in I[0]]
    context = "\n\n---\n\n".join(retrieved_chunks)

    # Create messages for OpenAI
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful financial assistant. "
                "Answer using ONLY the provided context. "
                "If the context is insufficient, say what is missing."
            ),
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    # Generate answer
    with st.spinner("Thinking..."):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2,
        )
        answer = response.choices[0].message.content

    # =========================
    # Display results
    # =========================
    st.markdown("### 🤖 Answer")
    st.markdown(answer)

    st.markdown("### 📚 Context Used")
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        preview = chunk[:220].replace("\n", " ")
        with st.expander(f"Chunk {idx}: {preview}..."):
            st.code(chunk, language="text")

