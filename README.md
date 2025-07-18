# 📊 GenAI Research Assistant for Capital Markets

A Streamlit-based app that uses **Retrieval-Augmented Generation (RAG)** to help analysts explore SEC filings (like 10-Ks). Ask questions and receive LLM-powered answers, backed by real source excerpts.
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://koreydillon-genai-research-assistant.streamlit.app/)


---

## 📌 Project Overview

This project demonstrates how to:
- Embed and search large financial documents using FAISS
- Use OpenAI’s GPT model to generate context-aware answers
- Build an interactive research assistant using Streamlit

---

## 🧠 Features

✅ LLM-powered Q&A using `gpt-3.5-turbo` or `gpt-4`  
✅ Embeds and indexes 10-K documents using `SentenceTransformers` + FAISS  
✅ Shows source excerpts used in each answer  
✅ Interactive Streamlit UI  
✅ Secure API key management with `.streamlit/secrets.toml`

---

## 🛠️ Tech Stack

- **Backend:** Python, FAISS, SentenceTransformers, OpenAI API
- **Frontend:** Streamlit
- **Data:** Public 10-K filings from [sec.gov](https://www.sec.gov/)

---

## 🚀 How to Run Locally

1. Clone this repo  
2. Set up a virtual environment and install requirements:
   ```bash
   pip install -r requirements.txt


## 💬 Sample Questions
“What risks did Microsoft list in their 2023 10-K?”
“Summarize Tesla’s competitive strategy.”
“What revenue segments are Apple focusing on?”

## 🖼️ Demo Preview Included

**Before input:**

[Start](assets/app.py.png)

**After input:**

[Answer](assets/user_input_&_answer.png)

