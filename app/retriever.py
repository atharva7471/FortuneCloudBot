import json
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"

COLLECTION_NAME = "fortune_cloud"

TOP_K_INITIAL = 5
TOP_K_FINAL = 2


# ============================================================
# LOAD VECTORSTORE
# ============================================================

def get_vectorstore():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
    )
    return vectorstore


# ============================================================
# QUERY INTENT DETECTION
# ============================================================

def detect_intent(query: str):
    """
    Detect what type of information the user is asking for.

    Returns:
        intent_name, metadata boosts
    """
    q = query.lower()

    # --------------------------------------------------------
    # OFFICE / BRANCH
    # --------------------------------------------------------
    if any(word in q for word in [
        "office", "branch", "address", "location", "located", 
        "phone number", "contact number", "telephone", "email address",
        "where is", "contact you",
    ]):
        return "office", {
            "office": 5.0,
            "office_directory": 3.0,
            "contact": 2.0,
        }

    # --------------------------------------------------------
    # COURSE CATALOG
    # --------------------------------------------------------
    if any(word in q for word in [
        "courses", "course list", "course catalog", "courses available",
        "what courses", "which courses", "available courses", "trainings available",
    ]):
        return "course_catalog", {
            "course_catalog": 6.0,
            "course": 2.0,
        }

    # --------------------------------------------------------
    # SPECIFIC COURSE
    # --------------------------------------------------------
    if any(word in q for word in [
        "duration", "course duration", "syllabus", "curriculum", 
        "course details", "course content", "training", "teach", "learn"
    ]) and "course" in q:
        return "course", {
            "course": 5.0,
            "course_catalog": 3.0,
        }

    # --------------------------------------------------------
    # BATCH SCHEDULE
    # --------------------------------------------------------
    if any(word in q for word in [
        "batch", "batch schedule", "batch timings", "batch timing", 
        "schedule", "next batch", "upcoming batch", "batch date", "start date",
    ]):
        return "batch_schedule", {
            "batch_schedule": 7.0,
            "course": 1.0,
        }

    # --------------------------------------------------------
    # REFUND / POLICY
    # --------------------------------------------------------
    if any(word in q for word in [
        "refund", "refund policy", "money back", "cancellation", 
        "cancel course", "fee refund",
    ]):
        return "refund_policy", {
            "policy": 6.0,
        }

    # --------------------------------------------------------
    # RECRUITERS
    # --------------------------------------------------------
    if any(word in q for word in [
        "recruiter", "recruiters", "hiring companies", "hiring partners", 
        "recruitment companies", "companies hiring", "placement companies",
    ]):
        return "recruiters", {
            "recruiters": 7.0,
            "success_stories": 2.0,
            "company": 1.0,
        }

    # --------------------------------------------------------
    # PLACEMENT
    # --------------------------------------------------------
    if any(word in q for word in [
        "placement", "placements", "job placement", "placement support", 
        "career support", "job assistance", "interview support",
    ]):
        return "placement", {
            "success_stories": 5.0,
            "company": 3.0,
            "careers": 2.0,
        }

    # --------------------------------------------------------
    # COMPANY / ABOUT
    # --------------------------------------------------------
    if any(word in q for word in [
        "who is", "about fortune cloud", "about fortune cloud technologies", 
        "company", "founder", "leadership", "ceo", "director", "history",
    ]):
        return "company", {
            "company": 6.0,
            "company_info": 5.0,
        }

    # --------------------------------------------------------
    # CAREERS
    # --------------------------------------------------------
    if any(word in q for word in [
        "career", "careers", "job opening", "job openings", 
        "vacancy", "vacancies", "work at fortune cloud", "join fortune cloud",
    ]):
        return "careers", {
            "careers": 7.0,
            "company": 2.0,
        }

    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------
    return "general", {}


# ============================================================
# RERANKING
# ============================================================

def rerank_results(query, results):
    """
    Re-rank Chroma results using semantic similarity + metadata.
    """
    intent, boosts = detect_intent(query)
    reranked = []

    for rank, doc in enumerate(results):
        metadata = doc.metadata or {}
        page_type = metadata.get("type", "webpage")

        # Base semantic score (earlier results receive a small advantage)
        semantic_score = max(0, TOP_K_INITIAL - rank) * 0.1
        metadata_boost = boosts.get(page_type, 0)
        
        category = str(metadata.get("category", "")).lower()
        category_boost = 0

        if intent == "refund_policy" and category == "refund":
            category_boost = 4.0
        elif intent == "company" and category in ["company", "about"]:
            category_boost = 3.0
        elif intent == "office" and page_type == "office":
            category_boost = 2.0
        elif intent == "course_catalog" and page_type == "course_catalog":
            category_boost = 2.0

        final_score = semantic_score + metadata_boost + category_boost

        reranked.append({
            "doc": doc,
            "rank": rank + 1,
            "page_type": page_type,
            "category": category,
            "semantic_score": semantic_score,
            "metadata_boost": metadata_boost,
            "category_boost": category_boost,
            "final_score": final_score,
        })

    # Highest score first
    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    return intent, reranked


# ============================================================
# HYBRID RETRIEVER SETUP
# ============================================================
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

DOCS_FILE = BASE_DIR / "data" / "raw" / "fortune_cloud_documents.json"

def get_bm25_retriever():
    if not DOCS_FILE.exists():
        return None
        
    with open(DOCS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    docs = [
        Document(page_content=item.get("content", ""), metadata=item.get("metadata", {})) 
        for item in data if item.get("content")
    ]
    
    if not docs:
        return None
        
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = TOP_K_INITIAL
    return bm25_retriever

vectorstore = get_vectorstore()
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_INITIAL})

bm25_retriever = get_bm25_retriever()

def reciprocal_rank_fusion(list1, list2, k=60):
    """
    Fuses two lists of ranked documents by assigning a score based on their rank.
    """
    fused_scores = {}
    doc_map = {}
    
    for doc_list in [list1, list2]:
        for rank, doc in enumerate(doc_list):
            doc_str = doc.page_content
            if doc_str not in doc_map:
                doc_map[doc_str] = doc
                fused_scores[doc_str] = 0.0
            # Higher rank (lower index) gives a higher score
            fused_scores[doc_str] += 1 / (rank + k)
            
    # Sort docs by fused score
    reranked_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_str] for doc_str, score in reranked_docs]


def get_context(query: str):
    """
    Retrieves via Hybrid Search (BM25 + Chroma), fuses them, and reranks based on intent.
    Returns just the LangChain Document objects.
    """
    # 1. Fetch from Vector Database
    results = vector_retriever.invoke(query)
    
    # 2. Fetch from BM25 Keyword Search
    if bm25_retriever:
        bm25_results = bm25_retriever.invoke(query)
        # Fuse the two lists together, prioritizing documents that appear highly in both
        results = reciprocal_rank_fusion(results, bm25_results)
        
    # 3. Apply Custom Business Logic Reranking
    _, reranked = rerank_results(query, results)
    
    # Return the top K Document objects
    return [item["doc"] for item in reranked[:TOP_K_FINAL]]
