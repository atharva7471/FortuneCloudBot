import os
import json
from pathlib import Path
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PINECONE_INDEX_NAME = "fortune-cloud"

TOP_K_INITIAL = 12
TOP_K_FINAL = 5


# ============================================================
# LOAD VECTORSTORE
# ============================================================

def get_vectorstore():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return None
    try:
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-mpnet-base-v2",
            huggingfacehub_api_token=hf_token
        )
        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings
        )
        return vectorstore
    except Exception:
        return None


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
    # OFFICE / BRANCH / ADDRESS / LOCATION
    # --------------------------------------------------------
    if any(word in q for word in [
        "office", "offices", "branch", "branches", "address", "addresses",
        "location", "locations", "located", "campus", "campuses",
        "phone number", "contact number", "telephone", "email address",
        "where is", "where are", "where do you", "contact you",
        "head office", "corporate office", "pimpri", "hadapsar",
        "shivajinagar", "bengaluru", "bangalore", "pune", "visit", "reach you"
    ]):
        return "office", {
            "office_directory": 10.0,
            "office": 6.0,
            "branch": 6.0,
            "contact": 4.0,
        }

    # --------------------------------------------------------
    # COURSE CATALOG / ENLIST COURSES
    # --------------------------------------------------------
    if any(word in q for word in [
        "enlist", "list courses", "list of courses", "all courses",
        "courses", "course list", "course catalog", "courses available",
        "what courses", "which courses", "available courses", "trainings available",
        "training programs", "offer", "curriculum list", "options"
    ]):
        return "course_catalog", {
            "course_catalog": 10.0,
            "course": 4.0,
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
        "who is", "who are you", "about fortune cloud", "about fortune cloud technologies", 
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
    Re-rank results using semantic similarity + section metadata.
    """
    intent, _ = detect_intent(query)
    reranked = []

    for rank, doc in enumerate(results):
        metadata = doc.metadata or {}
        section = metadata.get("section", "").lower()
        category = metadata.get("category", "").lower()

        # Base score giving rank preference
        semantic_score = max(0, TOP_K_INITIAL - rank) * 0.1
        metadata_boost = 0.0
        
        # Apply boosts based on the detected intent
        if intent == "office" and ("branch" in section or "location" in section or "address" in section or "office" in section or category == "office_directory"):
            metadata_boost += 10.0
        elif intent == "course_catalog" and ("course" in section or "catalog" in section or category == "course_catalog"):
            metadata_boost += 10.0
        elif intent == "course" and "course" in section:
            metadata_boost += 4.0
        elif intent == "batch_schedule" and ("batch" in section or "schedule" in section):
            metadata_boost += 5.0
        elif intent == "refund_policy" and ("refund" in section or "policy" in section or "legal" in section):
            metadata_boost += 5.0
        elif intent == "recruiters" and ("recruiter" in section or "placement" in section or "hire" in section):
            metadata_boost += 5.0
        elif intent == "placement" and ("placement" in section or "success" in section):
            metadata_boost += 5.0
        elif intent == "company" and ("overview" in section or "founder" in section or "about" in section):
            metadata_boost += 5.0
        elif intent == "careers" and ("career" in section or "work" in section):
            metadata_boost += 5.0

        final_score = semantic_score + metadata_boost

        reranked.append({
            "doc": doc,
            "rank": rank + 1,
            "section": section,
            "semantic_score": semantic_score,
            "metadata_boost": metadata_boost,
            "final_score": final_score,
        })

    # Highest score first
    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    return intent, reranked


# ============================================================
# HYBRID RETRIEVER SETUP
# ============================================================
import re
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

DOCS_FILE = BASE_DIR / "data" / "raw" / "fortune_cloud_documents.json"

STOPWORDS = {
    'do', 'does', 'did', 'you', 'your', 'know', 'the', 'a', 'an', 'is', 'are', 'was', 'were',
    'where', 'what', 'which', 'who', 'how', 'when', 'why', 'can', 'could', 'would', 'should',
    'tell', 'me', 'please', 'i', 'we', 'they', 'it', 'in', 'at', 'to', 'for', 'of', 'and', 'or',
    'this', 'that', 'with', 'about', 'have', 'has', 'had', 'any', 'some'
}

def preprocess_query(text: str):
    """
    Cleans punctuation and removes conversational stopwords for high-accuracy BM25 matching.
    """
    clean = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = [w for w in clean.split() if w not in STOPWORDS and len(w) > 1]
    return tokens if tokens else clean.split()

ALL_DOCS = []

def get_all_documents():
    global ALL_DOCS
    if not ALL_DOCS and DOCS_FILE.exists():
        with open(DOCS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        ALL_DOCS = [
            Document(page_content=item.get("content", ""), metadata=item.get("metadata", {}))
            for item in data if item.get("content")
        ]
    return ALL_DOCS

def get_bm25_retriever():
    docs = get_all_documents()
    if not docs:
        return None
        
    bm25 = BM25Retriever.from_documents(docs, preprocess_func=preprocess_query)
    bm25.k = TOP_K_INITIAL
    return bm25

try:
    vectorstore = get_vectorstore()
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_INITIAL}) if vectorstore else None
except Exception:
    vector_retriever = None

try:
    bm25_retriever = get_bm25_retriever()
except Exception:
    bm25_retriever = None

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
            fused_scores[doc_str] += 1 / (rank + k)
            
    reranked_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_str] for doc_str, score in reranked_docs]


def get_context(query: str):
    """
    Retrieves via Hybrid Search (BM25 + Pinecone), fuses them, and reranks based on intent.
    Guarantees inclusion of comprehensive directory context for office and course queries.
    """
    intent, _ = detect_intent(query)
    results = []
    
    # 1. Fetch from Vector Database (if available)
    if vector_retriever:
        try:
            results = vector_retriever.invoke(query)
        except Exception:
            results = []
    
    # 2. Fetch from BM25 Keyword Search
    if bm25_retriever:
        try:
            bm25_results = bm25_retriever.invoke(query)
            if results:
                results = reciprocal_rank_fusion(results, bm25_results)
            else:
                results = bm25_results
        except Exception:
            pass
        
    # 3. Apply Custom Business Logic Reranking
    _, reranked = rerank_results(query, results)
    final_docs = [item["doc"] for item in reranked[:TOP_K_FINAL]]
    
    # 4. Critical directory guarantee: ensure master directory chunks are present for key intents
    all_docs = get_all_documents()
    if intent == "office":
        for doc in all_docs:
            if "All Branch Locations" in doc.metadata.get("section", ""):
                if doc not in final_docs:
                    final_docs.insert(0, doc)
                break
    elif intent == "course_catalog":
        for doc in all_docs:
            if "Master Course Catalog" in doc.metadata.get("section", ""):
                if doc not in final_docs:
                    final_docs.insert(0, doc)
                break
                
    return final_docs[:TOP_K_FINAL]
