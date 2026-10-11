# Fortune Cloud Technologies — AI Virtual Assistant 🚀

A highly intelligent, fully cloud-native Retrieval-Augmented Generation (RAG) chatbot designed to act as a friendly, sales-oriented virtual assistant for Fortune Cloud Technologies. 

This project uses a modern Cloud RAG stack to answer student queries about courses, placements, schedules, and more, complete with accurate citations pointing back to the official website. It also features a **Dual-Mode Intelligent Form System** embedded seamlessly into a beautiful custom UI.

---

## 🏗️ Architecture & Tech Stack

This project uses a blazing fast **100% Serverless Cloud Architecture** combined with a custom-built Web Application frontend.

- **Backend:** FastAPI (Python)
- **Frontend:** Vanilla HTML, JavaScript, and Tailwind CSS
- **LLM Engine:** Groq API (`qwen/qwen3.8-27b` / `openai/gpt-oss-120b`) or Google Gemini (`gemini-3.5-flash`) via LangChain with automatic fallback
- **Vector Database:** Pinecone Serverless
- **Embeddings:** HuggingFace Inference API (`sentence-transformers/all-mpnet-base-v2`) / BM25
- **Data Orchestration:** LangChain (LCEL)

---

## 🌟 Key Features

1. **Intelligent Conversational AI:** Answers student queries accurately using RAG, directly citing the Fortune Cloud website.
2. **Server-Sent Events (SSE) Streaming:** Chat responses are streamed in real-time to the UI, providing a fast, ChatGPT-like experience.
3. **Dual-Mode Form System:** 
   - **Mode A (Conversational):** Collects simple details like name and phone number one-by-one inside the chat.
   - **Mode B (Embedded Forms):** Intelligently detects complex requests (e.g. course applications) and renders beautiful, fully interactive forms directly inside the chat window.
4. **Local Data Persistence:** Submitted forms are securely validated by FastAPI and appended locally to `data/enquiries.csv` for easy access by admissions teams.

---

## 🧠 How RAG Works in This Project

RAG (Retrieval-Augmented Generation) gives our AI model "open-book" access to your specific data. 

### 1. Data Ingestion (`app/crawler.py` & `app/ingest.py`)
* **The Crawler** reads `robots.txt` to find `llms-full.txt`—a perfectly structured Markdown representation of the website.
* **The Ingestor** splits the text intelligently by headers (`#`, `##`, `###`) using `MarkdownHeaderTextSplitter`.
* **Pinecone** index stores the embeddings created via HuggingFace.

### 2. Intelligent Retrieval (`app/retriever.py`)
* **Hybrid Search:** Searches Pinecone for "Semantic Meaning" and uses BM25 for "Keyword Matching".
* **Reciprocal Rank Fusion (RRF):** Mathematically merges both search results.
* **Intent-Based Reranking:** Analyzes the user's query and applies a **+5.0 Score Boost** to highly relevant chunks (e.g., boosting "course" sections if the user asks about courses).

### 3. Generation (`app/rag_chain.py`)
* The final chunks are injected into a carefully crafted **Prompt Template**.
* The prompt forces the LLM to act as a sales assistant, generate **Markdown Citations**, and dynamically generate **JSON Form Schemas** when a user wants to submit an enquiry.
* Features automatic fallback between Groq and Gemini.

---

## 🚀 Setup & Installation

### 1. Prerequisites
You need API keys for the cloud services powering the bot:
- **Groq (Recommended):** Free API key from [Groq Console](https://console.groq.com/) for blazing fast LLM inference with high free tier quota.
- **Google Gemini (Optional):** Free API key from [Google AI Studio](https://aistudio.google.com/) (`gemini-3.5-flash`).
- **Pinecone:** Free Serverless index named `fortune-cloud` (Dimensions: 768, Metric: cosine).
- **Hugging Face:** Free Inference API token for embeddings.

### 2. Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
HF_TOKEN=your_huggingface_token

# Optional LLM overrides
LLM_PROVIDER=groq
GROQ_MODEL=qwen/qwen3.8-27b
GEMINI_MODEL=gemini-3.5-flash
```

### 3. Running the Pipeline
To get the bot running, follow these steps in order:

```bash
# 1. Download the latest structured Markdown from the website
python app/crawler.py

# 2. Chunk, embed, and upload the data to Pinecone
python app/ingest.py

# 3. Launch the FastAPI Server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📂 Project Structure

```
FortuneCloudBot/
├── .env                        # Secret API keys
├── README.md                   # This file
├── data/                       # Stores the vector data and form submissions
│   ├── enquiries.csv           # Local storage for all submitted forms
│   └── raw/                    # Raw crawler data
├── templates/
│   └── index.html              # Custom Website + Chat UI
├── static/
│   └── js/
│       └── chat.js             # SSE Streaming and dynamic Form Rendering logic
└── app/
    ├── main.py                 # FastAPI Application Entrypoint
    ├── routes/                 # API Routes (chat.py, forms.py)
    ├── crawler.py              # Downloads data via robots.txt
    ├── ingest.py               # Splits and uploads data to Pinecone
    ├── retriever.py            # Advanced Hybrid Search + Reranking Engine
    └── rag_chain.py            # LangChain LCEL pipeline & Prompt definition
```
---

## 📧 Contact  
**Atharva Bhosale** 

📍 Pune, Maharashtra  
📩 Email: **atharva7471@gmail.com**
<br />  
🔗 LinkedIn: www.linkedin.com/in/atharvabhosale-ai
<br />
🐙 GitHub: https://github.com/atharva7471  
🌐 Portfolio: https://athoofolio.vercel.app/ 