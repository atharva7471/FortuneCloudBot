# Fortune Cloud Technologies — AI Virtual Assistant 🚀

A highly intelligent, fully cloud-native Retrieval-Augmented Generation (RAG) chatbot designed to act as a friendly, sales-oriented virtual assistant for Fortune Cloud Technologies. 

This project uses a modern Cloud RAG stack to answer student queries about courses, placements, schedules, and more, complete with accurate citations pointing back to the official website.

---

## 🏗️ Architecture & Tech Stack

This project was intentionally migrated from a local, heavy setup (Ollama/Chroma) to a blazing fast **100% Serverless Cloud Architecture**.

- **Frontend:** Streamlit (`app/ui.py`)
- **LLM Engine:** Groq API (`llama-3.3-70b-versatile`) via `langchain-groq`
- **Vector Database:** Pinecone Serverless
- **Embeddings:** HuggingFace Inference API (`sentence-transformers/all-mpnet-base-v2`)
- **Data Orchestration:** LangChain (LCEL)

---

## 🧠 How RAG Works in This Project

RAG (Retrieval-Augmented Generation) is a technique that gives an AI model "open-book" access to your specific data. Instead of training the model from scratch (which is expensive and gets outdated), we just *search* your data and pass the relevant paragraphs to the AI to read before it answers.

Here is the exact pipeline of how data flows through our system:

### 1. Data Ingestion (`app/crawler.py` & `app/ingest.py`)
Instead of blindly scraping messy HTML, our system respects modern AI web standards:
* **The Crawler** reads `robots.txt` to find `llms-full.txt`—a perfectly structured Markdown representation of the website.
* **The Ingestor** reads this Markdown and uses a `MarkdownHeaderTextSplitter`. Instead of cutting sentences in half, it splits the text intelligently by headers (`#`, `##`, `###`).
* It assigns precise **Metadata** to each chunk (e.g., `url: https://.../Success-Stories` and `section: Placements > Success Stories`).
* The chunks are then converted into numerical vectors (Embeddings) using HuggingFace and pushed to **Pinecone**, our cloud vector database.

### 2. Intelligent Retrieval (`app/retriever.py`)
When a user asks a question, we don't just do a simple search. We use an advanced multi-step retrieval engine:
* **Hybrid Search:** It searches Pinecone for "Semantic Meaning" (understanding concepts) and uses BM25 for "Keyword Matching" (exact words).
* **Reciprocal Rank Fusion (RRF):** It mathematically merges both search results so that chunks ranking high in *both* semantic and keyword searches bubble to the top.
* **Intent-Based Reranking:** The system analyzes the user's query (e.g., detecting they are asking about "Courses"). It then scans the retrieved chunks and applies a massive **+5.0 Score Boost** to any chunk whose metadata `section` contains the word "course". This ensures the LLM always gets the exact right context!

### 3. Generation (`app/rag_chain.py`)
* The final, highly-relevant chunks are formatted into a massive text block.
* They are injected into a carefully crafted **Prompt Template** alongside the Chat History and the User's Question.
* The prompt forces the Groq LLM to act as an enthusiastic sales assistant, and critically, it forces the LLM to generate **Markdown Citations** using the exact URL attached to the chunk's metadata!

---

## 🚀 Setup & Installation

### 1. Prerequisites
You need API keys for the three cloud services powering the bot:
- **Groq:** Free API key for the LLM.
- **Pinecone:** Free Serverless index named `fortune-cloud` (Dimensions: 768, Metric: cosine).
- **Hugging Face:** Free Inference API token for embeddings.

### 2. Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_key
PINECONE_API_KEY=your_pinecone_key
HF_TOKEN=your_huggingface_token
```

### 3. Running the Pipeline
To get the bot running from scratch, follow these steps in order:

```bash
# 1. Download the latest structured Markdown from the website
python app/crawler.py

# 2. Chunk, embed, and upload the data to Pinecone
python app/ingest.py

# 3. Launch the Chat UI
streamlit run app/ui.py
```

---

## 📂 Project Structure

```
FortuneCloudBot/
├── .env                        # Secret API keys
├── README.md                   # This file
├── data/
│   └── raw/
│       ├── fortune_cloud_llms_full.md    # The downloaded Markdown data
│       └── fortune_cloud_documents.json  # Fallback JSON used for BM25 search
└── app/
    ├── crawler.py              # Downloads data via robots.txt
    ├── ingest.py               # Splits and uploads data to Pinecone
    ├── retriever.py            # Advanced Hybrid Search + Reranking Engine
    ├── rag_chain.py            # LangChain LCEL pipeline & Prompt definition
    └── ui.py                   # Streamlit Frontend UI
```
---

## 📧 Contact  
**Atharva Bhosale** 

📍 Pune, Maharashtra  
📩 Email: **atharva7471@gmail.com**  
🔗 LinkedIn: **www.linkedin.com/in/atharvabhosale-ai**
🐙 GitHub: https://github.com/atharva7471  
🌐 Portfolio: https://athoofolio.vercel.app/ 
 