import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/raw/fortune_cloud_documents.json"
)

VECTORSTORE_DIR = Path(
    "data/vectorstore"
)

COLLECTION_NAME = "fortune_cloud"
EMBEDDING_MODEL = "nomic-embed-text"
BATCH_SIZE = 50

# ============================================================
# LOAD LANGCHAIN DOCUMENTS
# ============================================================

def load_documents():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    documents = []

    for item in data:

        content = item.get(
            "content",
            "",
        ).strip()

        metadata = item.get(
            "metadata",
            {},
        )

        if not content:
            continue

        documents.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )

    return documents


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("FORTUNE CLOUD — CHROMADB BUILD")
    print("=" * 72)

    print(
        f"\nInput file : {INPUT_FILE}"
    )

    print(
        f"Vectorstore: {VECTORSTORE_DIR}"
    )

    print(
        f"Collection  : {COLLECTION_NAME}"
    )

    print(
        f"Embedding   : {EMBEDDING_MODEL}"
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    documents = load_documents()

    print(
        f"\nDocuments loaded: "
        f"{len(documents)}"
    )

    if not documents:
        raise RuntimeError(
            "No documents found."
        )

    # --------------------------------------------------------
    # SHOW SAMPLE METADATA
    # --------------------------------------------------------

    print("\nSample metadata:")

    for key, value in (
        documents[0].metadata.items()
    ):

        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    print(
        "\nInitializing Ollama embeddings..."
    )

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # CHROMA
    # --------------------------------------------------------

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(
            VECTORSTORE_DIR
        ),
    )

    # --------------------------------------------------------
    # ADD DOCUMENTS IN BATCHES
    # --------------------------------------------------------

    total = len(documents)

    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            total,
        )

        batch = documents[
            start:end
        ]

        print(
            f"\nAdding documents "
            f"{start + 1}-{end} "
            f"of {total}..."
        )

        vectorstore.add_documents(
            batch
        )

        print(
            f"  Added: {len(batch)}"
        )

    # --------------------------------------------------------
    # VERIFY
    # --------------------------------------------------------

    try:

        collection_count = (
            vectorstore
            ._collection
            .count()
        )

    except Exception:

        collection_count = None

    print(
        "\n" + "=" * 72
    )

    print(
        "CHROMADB BUILD COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Expected documents : {total}"
    )

    if collection_count is not None:

        print(
            f"Chroma documents   : "
            f"{collection_count}"
        )

        if collection_count == total:

            print(
                "\n✓ Document count verified."
            )

        else:

            print(
                "\n⚠ Document count mismatch!"
            )

    print(
        f"\nVectorstore saved to:"
        f"\n{VECTORSTORE_DIR}"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()