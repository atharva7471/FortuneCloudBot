import json
from pathlib import Path
import shutil
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_MD = BASE_DIR / "data" / "raw" / "fortune_cloud_llms_full.md"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"
OUTPUT_JSON = BASE_DIR / "data" / "raw" / "fortune_cloud_documents.json" 

def main():
    if not INPUT_MD.exists():
        print(f"File not found: {INPUT_MD}")
        print("Please run crawler.py first!")
        return

    print("=" * 60)
    print("FORTUNE CLOUD — INTELLIGENT INGESTION")
    print("=" * 60)

    with open(INPUT_MD, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # Intelligently split the markdown based on its native Header structure
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
    
    for split in md_header_splits:
        content = split.page_content
        metadata = split.metadata
        
        # Extract the URL if it's explicitly written in the chunk
        url = "https://www.fortunecloudindia.com"
        for line in content.splitlines():
            if "**URL**:" in line:
                url = line.split("**URL**:")[1].strip()
                break
            elif "**Canonical URL**:" in line:
                url = line.split("**Canonical URL**:")[1].strip()
                break
                
        metadata["url"] = url
        
        # Create a breadcrumb trail (e.g. "Courses > Technical > Data Science")
        h_path = " > ".join(v for k, v in metadata.items() if k.startswith("Header"))
        metadata["section"] = h_path
        
        docs_for_json.append({
            "content": content,
            "metadata": metadata
        })
        
        final_docs.append(Document(page_content=content, metadata=metadata))

    # 1. Save JSON for the BM25 Retriever
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(docs_for_json, f, ensure_ascii=False, indent=2)

    print(f"Split into {len(final_docs)} perfectly structured chunks.")

    # 2. Rebuild the Chroma Vector Database
    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)
        
    print("Embedding chunks using Nomic... (this may take a moment)")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    Chroma.from_documents(
        documents=final_docs,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR)
    )
    
    print("Vectorstore rebuilt successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()