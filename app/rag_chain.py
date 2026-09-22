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

PROMPT_TEMPLATE = """You are a friendly, human-like virtual assistant for Fortune Cloud Technologies.
Your goal is to help users by answering their questions in a natural, conversational, and welcoming tone.

Rules:
1. Be conversational and friendly. Do not sound like a robot.
2. Answer the user's question using the provided Context, but DO NOT use phrases like "Based on the provided context" or "According to the context". Just give the answer naturally as if you know it.
3. If the answer cannot be found in the Context, politely apologize and say you don't have that information.
4. Do not make up or guess any information.
5. Use the Chat History to understand context for follow-up questions.
6. **IMPORTANT**: Whenever you provide factual information from the Context, you MUST add a markdown link to the source at the end of the sentence or paragraph (e.g. `[Source](url)`). Use the URL provided in the `[Source: ...]` blocks.

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
        source = doc.metadata.get("source", "Unknown")
        # Format the text so the LLM knows the source of this specific chunk
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
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

