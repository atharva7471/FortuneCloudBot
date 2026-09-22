import os
from dotenv import load_dotenv, find_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from retriever import get_context

# Load environment variables from .env file
load_dotenv()

# ============================================================
# INITIALIZE LLM
# ============================================================

# Initialize the Groq model
llm = ChatGroq(
    model="openai/gpt-oss-120b", 
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

# ============================================================
# PROMPT TEMPLATE
# ============================================================

PROMPT_TEMPLATE = """You are the official Virtual Assistant for Fortune Cloud Technologies, India's leading AI-integrated IT training institute.
Your goal is to enthusiastically help prospective students by answering their questions in a natural, conversational, and welcoming tone.

Rules:
1. **Be Enthusiastic and Sales-Oriented**: Be highly encouraging. Emphasize that our courses are 100% job-oriented and focus on our 94% placement rate and real-world projects.
2. **Be Conversational**: Do not sound like a robot. Do not use phrases like "Based on the provided context" or "According to the context".
3. **Handle Missing Info Gracefully**: If the answer cannot be found in the Context, politely say you don't have that specific information, and encourage them to reach out to our admissions team at info@fortunecloudindia.com or call +91-9766439090. Do not guess.
4. **Use Chat History**: Refer to the Chat History to understand context for follow-up questions.
5. **Formatting**: Use markdown to make your response easy to read (bullet points, bold text for emphasis).
6. **Citations (CRITICAL)**: Whenever you provide factual information from the Context, you MUST add a markdown link to the source at the end of the sentence or paragraph (e.g. `[Source](url)`). Use the exact URL provided in the `[Source URL: ...]` blocks.

Chat History:
{chat_history}

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


# ============================================================
# CHAIN DEFINITION
# ============================================================

def format_docs(docs):
    """
    Combine the page_content of all retrieved documents into a single string.
    Injects the source metadata so the LLM can cite it.
    """
    formatted = []
    for doc in docs:
        source = doc.metadata.get("url", "https://www.fortunecloudindia.com")
        # Format the text so the LLM knows the source of this specific chunk
        formatted.append(f"[Source URL: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


# The RAG chain uses LangChain Expression Language (LCEL)
rag_chain = (
    # We expect a dict like {"question": "...", "chat_history": "..."}
    RunnablePassthrough.assign(
        context=lambda x: format_docs(get_context(x["question"]))
    )
    | prompt
    | llm
    | StrOutputParser()
)

def ask_qwen(question: str, chat_history: str = ""):
    """
    Helper function to invoke the chain with history.
    """
    return rag_chain.invoke({"question": question, "chat_history": chat_history})

def stream_qwen(question: str, chat_history: str = ""):
    """
    Helper function to stream the chain output with history.
    """
    return rag_chain.stream({"question": question, "chat_history": chat_history})

