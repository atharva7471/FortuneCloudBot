import json
import re
from pathlib import Path
from collections import Counter

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CRAWL_FILE = BASE_DIR / "data" / "raw" / "fortune_cloud_crawl4ai.json"
OFFICES_FILE = BASE_DIR / "data" / "raw" / "offices.json"
OUTPUT_FILE = BASE_DIR / "data" / "raw" / "fortune_cloud_documents.json"


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# URL FILTERING
# ============================================================

def is_lms_url(url: str) -> bool:
    return "lms.fortunecloudindia.com" in url.lower()


def is_invalid_url(url: str) -> bool:
    return "/ontact" in url.lower()


def is_contact_page(url: str) -> bool:
    return "/contact" in url.lower()


def is_courses_page(url: str) -> bool:
    path = url.lower().rstrip("/")

    return path.endswith("/courses")


# ============================================================
# REMOVE REPEATED FOOTER
# ============================================================

def remove_repeated_contact_section(text: str, url: str) -> str:

    # Keep complete contact page
    if is_contact_page(url):
        return text

    markers = [
        "### Reach Us Directly",
        "## Reach Us Directly",
        "# Reach Us Directly",
        "Reach Us Directly",
    ]

    positions = []

    for marker in markers:

        position = text.lower().find(marker.lower())

        if position != -1:
            positions.append(position)

    if positions:

        cutoff = min(positions)

        text = text[:cutoff]

    return text.strip()


# ============================================================
# CLEAN MARKDOWN
# ============================================================

def clean_markdown(text: str, url: str) -> str:

    if not text:
        return ""

    # Remove repeated footer
    text = remove_repeated_contact_section(text, url)

    # Convert markdown links to visible text
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text
    )

    # Remove HTML links
    text = re.sub(
        r"<a\b[^>]*>(.*?)</a>",
        r"\1",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    noise_lines = {
        "ADMISSIONS OPEN - LEARN, GROW & LEAD",
        "Admissions Open",
        "Call +91-9766439090",
        "Batch Schedule",
        "Facebook",
        "Instagram",
        "LinkedIn",
        "Twitter",
        "YouTube",
    }

    cleaned_lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if line in noise_lines:
            continue

        # Remove social media URLs
        if re.match(
            r"^https?://(www\.)?(facebook|instagram|linkedin|twitter|youtube)\.",
            line,
            flags=re.IGNORECASE,
        ):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Remove repeated short lines
    final_lines = []

    seen_short_lines = set()

    for line in text.splitlines():

        normalized = re.sub(
            r"\s+",
            " ",
            line
        ).strip()

        if not normalized:
            continue

        if len(normalized) < 100:

            key = normalized.lower()

            if key in seen_short_lines:
                continue

            seen_short_lines.add(key)

        final_lines.append(normalized)

    text = "\n".join(final_lines)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# CLASSIFY WEBSITE PAGE
# ============================================================

def classify_page(url: str, title: str):

    url_lower = url.lower()
    title_lower = title.lower()

    # --------------------------------------------------------
    # COURSE CATALOG
    # --------------------------------------------------------

    if is_courses_page(url):

        return "course_catalog", "training"

    # --------------------------------------------------------
    # COURSE PAGES
    # --------------------------------------------------------

    course_keywords = [
        "course",
        "training",
        "master-in-",
        "ai-powered-",
        "edge-java",
        "edge-python",
        "edge-dotnet",
        "sap-",
        "software-testing-with-ai",
        "certified-hr",
        "digital-marketing",
    ]

    if any(
        keyword in url_lower or keyword in title_lower
        for keyword in course_keywords
    ):

        return "course", "training"

    # --------------------------------------------------------
    # ABOUT / COMPANY
    # --------------------------------------------------------

    if (
        "/about-us" in url_lower
        or "about us" in title_lower
        or "about fortune" in title_lower
    ):

        return "company", "about"

    # --------------------------------------------------------
    # CONTACT
    # --------------------------------------------------------

    if is_contact_page(url):

        return "contact", "contact"

    # --------------------------------------------------------
    # RECRUITERS
    # --------------------------------------------------------

    if "/recruiters" in url_lower:

        return "recruiters", "placement"

    # --------------------------------------------------------
    # SUCCESS STORIES
    # --------------------------------------------------------

    if (
        "success" in url_lower
        or "success stories" in title_lower
        or "placement" in title_lower
    ):

        return "success_stories", "placement"

    # --------------------------------------------------------
    # BATCH SCHEDULE
    # --------------------------------------------------------

    if (
        "batchschedule" in url_lower
        or "batch schedule" in title_lower
    ):

        return "batch_schedule", "admissions"

    # --------------------------------------------------------
    # CAREERS
    # --------------------------------------------------------

    if (
        "/careers" in url_lower
        or "career" in title_lower
    ):

        return "careers", "company"

    # --------------------------------------------------------
    # FRANCHISE
    # --------------------------------------------------------

    if (
        "franchise" in url_lower
        or "franchise" in title_lower
    ):

        return "franchise", "business"

    # --------------------------------------------------------
    # CAMPUS / EVENTS
    # --------------------------------------------------------

    if (
        "open-campus" in url_lower
        or "campus drive" in title_lower
        or "event" in title_lower
    ):

        return "event", "campus_drive"

    # --------------------------------------------------------
    # BLOG
    # --------------------------------------------------------

    if (
        "/blog" in url_lower
        or "blog" in title_lower
    ):

        return "blog", "resources"

    # --------------------------------------------------------
    # POLICIES
    # --------------------------------------------------------

    if (
        "privacy" in url_lower
        or "privacy" in title_lower
    ):

        return "policy", "privacy"

    if (
        "refund" in url_lower
        or "refund" in title_lower
    ):

        return "policy", "refund"

    if (
        "profanity" in url_lower
        or "profanity" in title_lower
    ):

        return "policy", "profanity"

    if (
        "terms" in url_lower
        or "terms" in title_lower
    ):

        return "policy", "terms"

    # --------------------------------------------------------
    # QUICK INFO
    # --------------------------------------------------------

    if (
        "quickinfo" in url_lower
        or "quick info" in title_lower
    ):

        return "company_info", "general"

    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return "webpage", "general"


# ============================================================
# CREATE STRUCTURED OFFICE DOCUMENTS
# ============================================================

def create_office_documents(offices):

    documents = []

    directory_parts = [
        "Fortune Cloud Technologies Office Directory",
        "",
    ]

    for office in offices:

        office_name = office.get(
            "office",
            "Unknown Office"
        )

        raw_content = office.get(
            "content",
            ""
        )

        source = office.get(
            "source",
            ""
        )

        # ----------------------------------------------------
        # Clean office content
        # ----------------------------------------------------

        content = re.sub(
            r"\s+",
            " ",
            raw_content
        ).strip()

        # Remove unnecessary UI text
        content = re.sub(
            r"\bGOOGLE REVIEWS\b.*?\bREVIEWS\b",
            "",
            content,
            flags=re.IGNORECASE
        )

        content = re.sub(
            r"\bVIEW MAP\s*→?",
            "",
            content,
            flags=re.IGNORECASE
        )

        content = re.sub(
            r"\s+",
            " ",
            content
        ).strip()

        # ----------------------------------------------------
        # Individual office document
        # ----------------------------------------------------

        office_content = (
            f"Office: {office_name}\n\n"
            f"{content}"
        )

        documents.append(
            Document(
                page_content=office_content,
                metadata={
                    "source": source,
                    "url": source,
                    "title": office_name,
                    "type": "office",
                    "category": "contact",
                    "crawler": "structured",
                    "structured": True,
                },
            )
        )

        # ----------------------------------------------------
        # Directory
        # ----------------------------------------------------

        directory_parts.append(
            f"Office: {office_name}"
        )

        directory_parts.append(
            content
        )

        directory_parts.append("")

    # ========================================================
    # COMPLETE OFFICE DIRECTORY
    # ========================================================

    directory_content = "\n".join(
        directory_parts
    )

    documents.append(
        Document(
            page_content=directory_content,
            metadata={
                "source": "structured_office_data",
                "url": "",
                "title": "Fortune Cloud Technologies Office Directory",
                "type": "office_directory",
                "category": "contact",
                "crawler": "structured",
                "structured": True,
            },
        )
    )

    return documents


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("FORTUNE CLOUD — LANGCHAIN STRUCTURED INGESTION")
    print("=" * 72)

    print(f"\nInput : {CRAWL_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    crawl_data = load_json(
        CRAWL_FILE
    )

    offices = load_json(
        OFFICES_FILE
    )

    print(
        f"\nCrawl4AI documents loaded: "
        f"{len(crawl_data)}"
    )

    print(
        f"Structured offices loaded : "
        f"{len(offices)}"
    )

    # --------------------------------------------------------
    # Website documents
    # --------------------------------------------------------

    documents = []

    skipped = []

    original_characters = 0
    cleaned_characters = 0

    for item in crawl_data:

        url = item.get(
            "url",
            ""
        )

        title = item.get(
            "title",
            ""
        )

        content = item.get(
            "content",
            ""
        )

        if not url:
            continue

        # ----------------------------------------------
        # Filter LMS
        # ----------------------------------------------

        if is_lms_url(url):

            skipped.append(
                (url, "filtered_url")
            )

            continue

        # ----------------------------------------------
        # Filter malformed URL
        # ----------------------------------------------

        if is_invalid_url(url):

            skipped.append(
                (url, "filtered_url")
            )

            continue

        original_characters += len(
            content
        )

        # ----------------------------------------------
        # Clean
        # ----------------------------------------------

        cleaned = clean_markdown(
            content,
            url
        )

        cleaned_characters += len(
            cleaned
        )

        if len(cleaned) < 50:

            skipped.append(
                (url, "too_short")
            )

            continue

        # ----------------------------------------------
        # Classification
        # ----------------------------------------------

        page_type, category = classify_page(
            url,
            title
        )

        metadata = {
            "source": url,
            "url": url,
            "title": title,
            "type": page_type,
            "category": category,
            "crawler": "crawl4ai",
            "structured": False,
        }

        documents.append(
            Document(
                page_content=cleaned,
                metadata=metadata,
            )
        )

    # --------------------------------------------------------
    # Add structured office knowledge
    # --------------------------------------------------------

    office_documents = create_office_documents(
        offices
    )

    documents.extend(
        office_documents
    )

    # --------------------------------------------------------
    # Document count
    # --------------------------------------------------------

    print(
        f"\nLangChain documents created: "
        f"{len(documents)}"
    )

    print(
        f"Documents skipped: "
        f"{len(skipped)}"
    )

    # --------------------------------------------------------
    # Document type summary
    # --------------------------------------------------------

    type_counter = Counter(
        doc.metadata.get(
            "type",
            "unknown"
        )
        for doc in documents
    )

    print("\nDocument types:")

    for doc_type, count in sorted(
        type_counter.items()
    ):

        print(
            f"  {doc_type}: {count}"
        )

    # --------------------------------------------------------
    # Cleaning statistics
    # --------------------------------------------------------

    removed = (
        original_characters
        - cleaned_characters
    )

    if original_characters:

        reduction = (
            removed
            / original_characters
            * 100
        )

    else:

        reduction = 0

    print("\nCleaning statistics:")

    print(
        f"  Original characters : "
        f"{original_characters:,}"
    )

    print(
        f"  Cleaned characters  : "
        f"{cleaned_characters:,}"
    )

    print(
        f"  Removed characters  : "
        f"{removed:,}"
    )

    print(
        f"  Reduction           : "
        f"{reduction:.2f}%"
    )

    # ========================================================
    # CHUNKING
    # ========================================================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n### ",
            "\n## ",
            "\n# ",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(
        documents
    )

    # --------------------------------------------------------
    # Chunk IDs
    # --------------------------------------------------------

    for index, chunk in enumerate(
        chunks
    ):

        chunk.metadata[
            "chunk_id"
        ] = index

    # ========================================================
    # SAVE
    # ========================================================

    output = []

    for chunk in chunks:

        output.append(
            {
                "content": chunk.page_content,
                "metadata": chunk.metadata,
            }
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    total_chunk_characters = sum(
        len(chunk.page_content)
        for chunk in chunks
    )

    print(
        f"\nFinal chunks: "
        f"{len(chunks):,}"
    )

    print(
        f"Total chunk characters: "
        f"{total_chunk_characters:,}"
    )

    print(
        f"\nSaved to: "
        f"{OUTPUT_FILE}"
    )

    # ========================================================
    # SKIPPED DOCUMENTS
    # ========================================================

    if skipped:

        print("\nSkipped documents:")

        for url, reason in skipped:

            print(
                f"  - {url} "
                f"({reason})"
            )

    print("\n" + "=" * 72)
    print("STRUCTURED INGESTION COMPLETE")
    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()