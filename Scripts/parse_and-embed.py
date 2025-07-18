import os
import re
import openai
import faiss
import pickle
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# === CONFIG ===
FILENAME = "../Data/Microsoft_10-K_2023.html"
CHUNK_SIZE = 500  # characters
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SBERT model
VECTOR_DB_PATH = "vector_store/faiss_index"
EMBEDDINGS_PATH = "vector_store/chunks.pkl"

# === STEP 1: Load & Clean ===
with open(FILENAME, "r", encoding="latin1") as f:
    soup = BeautifulSoup(f.read(), "html.parser")

# Remove scripts/styles and clean text
for tag in soup(["script", "style"]): tag.decompose()
text = soup.get_text()
text = re.sub(r'\s+', ' ', text)

# === STEP 2: Chunking ===
chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

# === STEP 3: Create Embeddings ===
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(chunks)

# === STEP 4: Store in FAISS ===
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

os.makedirs("vector_store", exist_ok=True)
faiss.write_index(index, VECTOR_DB_PATH)
with open(EMBEDDINGS_PATH, "wb") as f:
    pickle.dump(chunks, f)

print("Vector store created and saved.")
