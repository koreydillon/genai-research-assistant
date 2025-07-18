# app.py

import pickle
import faiss
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# === Load API key from Streamlit secrets ===
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# === Load vector store and chunks ===
index = faiss.read_index("vector_store/faiss_index")
with open("vector_store/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# === Load embedding model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Streamlit UI ===
st.title("📊 SEC 10-K GenAI Assistant")
query = st.text_input("Ask a question about the document:")

if query:
    # Get embedding for query
    query_embedding = model.encode([query])
    _, I = index.search(query_embedding, k=3)

    # Retrieve top chunks
    retrieved_chunks = [chunks[i] for i in I[0]]
    context = "\n\n".join(retrieved_chunks)

    # Create messages for OpenAI
    messages = [
        {"role": "system", "content": "You are a helpful financial assistant. Use the context below to answer the user's question."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    # Generate answer using OpenAI
    with st.spinner("Thinking..."):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2
        )
        answer = response.choices[0].message.content

    # Display the results
    st.markdown("### 🤖 Answer")
    st.write(answer)

    st.markdown("### 📚 Context Used")
    for chunk in retrieved_chunks:
        st.markdown(f"> {chunk[:500]}...\n")
