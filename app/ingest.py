import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_MD = BASE_DIR / "data" / "raw" / "fortune_cloud_llms_full.md"
OUTPUT_JSON = BASE_DIR / "data" / "raw" / "fortune_cloud_documents.json" 

PINECONE_INDEX_NAME = "fortune-cloud"

def main():
    if not INPUT_MD.exists():
        print(f"File not found: {INPUT_MD}")
        print("Please run crawler.py first!")
        return

    print("=" * 60)
    print("FORTUNE CLOUD — CLOUD INGESTION (PINECONE & HF)")
    print("=" * 60)

    with open(INPUT_MD, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_text)

    docs_for_json = []
    final_docs = []
    
    current_url = "https://www.fortunecloudindia.com"
    
    for split in md_header_splits:
        content = split.page_content
        metadata = split.metadata
        
        for line in content.splitlines():
            if "**URL**:" in line:
                current_url = line.split("**URL**:")[1].strip()
                break
            elif "**Canonical URL**:" in line:
                current_url = line.split("**Canonical URL**:")[1].strip()
                break
                
        metadata["url"] = current_url
        h_path = " > ".join(v for k, v in metadata.items() if k.startswith("Header"))
        metadata["section"] = h_path
        
        docs_for_json.append({
            "content": content,
            "metadata": metadata
        })
        final_docs.append(Document(page_content=content, metadata=metadata))

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(docs_for_json, f, ensure_ascii=False, indent=2)

    print(f"Split into {len(final_docs)} chunks.")

    print("Initializing HuggingFace Embeddings (Cloud)...")
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-mpnet-base-v2",
        huggingfacehub_api_token=os.environ.get("HF_TOKEN")
    )
    
    print(f"Uploading to Pinecone index: {PINECONE_INDEX_NAME}...")
    PineconeVectorStore.from_documents(
        documents=final_docs,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    
    print("Upload complete! You are now fully migrated to the cloud.")
    print("=" * 60)

if __name__ == "__main__":
    main()