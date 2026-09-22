import asyncio
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse, urljoin
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
)
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://www.fortunecloudindia.com/"

OUTPUT_FILE = Path(
    "data/raw/fortune_cloud_crawl4ai.json"
)

DISCOVERY_FILE = Path(
    "data/raw/fortune_cloud_discovery.json"
)


# ============================================================
# DOMAIN CONFIGURATION
# ============================================================

ROOT_DOMAIN = "fortunecloudindia.com"


# ============================================================
# FILE TYPES
# ============================================================

FILE_EXTENSIONS = {
    ".pdf": "pdf",
    ".doc": "document",
    ".docx": "document",
    ".xls": "spreadsheet",
    ".xlsx": "spreadsheet",
    ".csv": "csv",
    ".zip": "archive",
    ".rar": "archive",
    ".apk": "apk",
    ".exe": "executable",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".gif": "image",
    ".svg": "image",
    ".mp4": "video",
    ".webm": "video",
}


# ============================================================
# HELPERS
# ============================================================

def content_hash(text: str) -> str:
    """
    Generate SHA256 hash for content.
    Useful later for detecting page changes.
    """

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def normalize_url(url: str) -> str:
    """
    Normalize URLs so the same page isn't stored multiple times.
    """

    if not url:
        return ""

    url = url.strip()

    parsed = urlparse(url)

    # Remove fragment:
    # /courses#python
    # becomes
    # /courses
    normalized = parsed._replace(
        fragment=""
    ).geturl()

    return normalized.rstrip("/")


def get_hostname(url: str) -> str:
    """
    Return lowercase hostname.
    """

    try:
        return (
            urlparse(url)
            .hostname
            or ""
        ).lower()
    except Exception:
        return ""


def is_root_domain(url: str) -> bool:
    """
    True only for:

        fortunecloudindia.com
        www.fortunecloudindia.com
    """

    hostname = get_hostname(url)

    return hostname in {
        ROOT_DOMAIN,
        f"www.{ROOT_DOMAIN}",
    }


def is_subdomain(url: str) -> bool:
    """
    Detect subdomains such as:

        lms.fortunecloudindia.com
        testdash.fortunecloudindia.com
    """

    hostname = get_hostname(url)

    if not hostname:
        return False

    suffix = "." + ROOT_DOMAIN

    return (
        hostname.endswith(suffix)
        and hostname != ROOT_DOMAIN
    )


def is_fortune_cloud_domain(url: str) -> bool:
    """
    True for root domain OR any subdomain.
    """

    return (
        is_root_domain(url)
        or is_subdomain(url)
    )


def classify_domain(url: str) -> str:
    """
    Classify URL based on hostname.
    """

    if is_root_domain(url):
        return "root_domain"

    if is_subdomain(url):
        return "subdomain"

    return "external"


def get_extension(url: str) -> str:
    """
    Return lowercase file extension.
    """

    path = urlparse(url).path.lower()

    if "." not in path:
        return ""

    return Path(path).suffix.lower()


def classify_resource(url: str) -> str:
    """
    Classify the discovered resource.

    Examples:

        /courses
            -> html

        /certificate.pdf
            -> pdf

        /FC-LMS.apk
            -> apk
    """

    extension = get_extension(url)

    if extension in FILE_EXTENSIONS:
        return FILE_EXTENSIONS[extension]

    return "html"


# ============================================================
# URL CLASSIFICATION
# ============================================================

def classify_url(url: str) -> dict:
    """
    Produce a complete classification record for a URL.
    """

    url = normalize_url(url)

    return {
        "url": url,
        "domain_type": classify_domain(url),
        "resource_type": classify_resource(url),
        "hostname": get_hostname(url),
        "extension": get_extension(url),
    }


# ============================================================
# CRAWL RESULT → DOCUMENT
# ============================================================

def result_to_document(result, depth=None):
    """
    Convert Crawl4AI CrawlResult into our standard document format.
    """

    markdown = result.markdown

    # Some Crawl4AI versions expose markdown as an object.
    if hasattr(markdown, "raw_markdown"):
        markdown = markdown.raw_markdown

    markdown = markdown or ""

    metadata = result.metadata or {}

    links = result.links or {}

    internal_links = (
        links.get("internal", [])
        or []
    )

    external_links = (
        links.get("external", [])
        or []
    )

    discovered_links = []

    for link in internal_links:

        if isinstance(link, dict):
            href = (
                link.get("href")
                or link.get("url")
            )
        else:
            href = str(link)

        if not href:
            continue

        href = normalize_url(
            urljoin(result.url, href)
        )

        if not href:
            continue

        discovered_links.append(
            href
        )

    return {
        "url": normalize_url(
            result.url
        ),

        "title": (
            metadata.get("title")
            or ""
        ),

        "content": markdown,

        "content_format": "markdown",

        "source": "crawl4ai",

        "success": result.success,

        "status_code": result.status_code,

        "content_hash": content_hash(
            markdown
        ),

        "domain_type": classify_domain(
            result.url
        ),

        "resource_type": classify_resource(
            result.url
        ),

        "crawl_depth": depth,

        "internal_links": sorted(
            set(discovered_links)
        ),

        "internal_link_count": len(
            discovered_links
        ),

        "external_link_count": len(
            external_links
        ),

        "metadata": metadata,
    }


# ============================================================
# DISCOVERY RECORD
# ============================================================

def create_discovery_record(
    url,
    discovered_from=None,
    depth=None,
    status="discovered",
):
    """
    Create lightweight URL inventory record.
    """

    info = classify_url(url)

    return {
        "url": info["url"],

        "hostname": info["hostname"],

        "domain_type": info[
            "domain_type"
        ],

        "resource_type": info[
            "resource_type"
        ],

        "extension": info[
            "extension"
        ],

        "discovered_from": (
            discovered_from
            or ""
        ),

        "depth": depth,
        "status": status,
    }


# ============================================================
# MAIN CRAWLER
# ============================================================

async def crawl_fortune_cloud(
    max_depth: int = 5,
    max_pages: int = 200,
):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
    )

    # --------------------------------------------------------
    # BFS DEEP CRAWLER
    # --------------------------------------------------------

    deep_crawl = BFSDeepCrawlStrategy(
        max_depth=max_depth,
        max_pages=max_pages,

        # We will classify external/subdomain URLs
        # ourselves.
        include_external=True,
    )

    crawl_config = CrawlerRunConfig(
        deep_crawl_strategy=deep_crawl,

        cache_mode=CacheMode.BYPASS,

        stream=True,

        wait_until="networkidle",
    )

    documents = []

    discovery = []

    seen_urls = set()

    print("=" * 72)
    print(
        "FORTUNE CLOUD — CRAWL4AI WEBSITE DISCOVERY"
    )
    print("=" * 72)

    print(
        f"Start URL : {BASE_URL}"
    )

    print(
        f"Max depth : {max_depth}"
    )

    print(
        f"Max pages : {max_pages}"
    )

    print()

    # --------------------------------------------------------
    # START CRAWLER
    # --------------------------------------------------------

    async with AsyncWebCrawler(
        config=browser_config
    ) as crawler:

        results = await crawler.arun(
            url=BASE_URL,
            config=crawl_config,
        )

        async for result in results:

            url = normalize_url(
                result.url
            )

            # ------------------------------------------------
            # DUPLICATE URL
            # ------------------------------------------------

            if url in seen_urls:
                continue

            seen_urls.add(url)

            # ------------------------------------------------
            # CLASSIFY URL
            # ------------------------------------------------

            info = classify_url(url)

            print(
                f"\n[DISCOVERED]"
            )

            print(
                f"URL      : {url}"
            )

            print(
                f"Domain   : {info['domain_type']}"
            )

            print(
                f"Resource : {info['resource_type']}"
            )

            # ------------------------------------------------
            # NON-HTML RESOURCE
            # ------------------------------------------------

            if info[
                "resource_type"
            ] != "html":

                print(
                    "Action   : INVENTORY ONLY"
                )

                discovery.append(
                    create_discovery_record(
                        url=url,
                        depth=None,
                        status="non_html_resource",
                    )
                )

                continue

            # ------------------------------------------------
            # FAILED HTML PAGE
            # ------------------------------------------------

            if not result.success:

                print(
                    "Status   : FAILED"
                )

                discovery.append(
                    create_discovery_record(
                        url=url,
                        status="failed",
                    )
                )

                continue

            # ------------------------------------------------
            # CONVERT TO DOCUMENT
            # ------------------------------------------------

            document = result_to_document(
                result
            )

            # ------------------------------------------------
            # VERY SMALL CONTENT
            # ------------------------------------------------

            if len(
                document[
                    "content"
                ].strip()
            ) < 100:

                print(
                    "Action   : SKIPPED "
                    "(very little content)"
                )

                discovery.append(
                    create_discovery_record(
                        url=url,
                        status="low_content",
                    )
                )

                continue

            # ------------------------------------------------
            # SAVE DOCUMENT
            # ------------------------------------------------

            documents.append(
                document
            )

            discovery.append(
                create_discovery_record(
                    url=url,
                    status="crawled",
                )
            )

            print(
                f"Action   : CRAWLED"
            )

            print(
                f"Content  : "
                f"{len(document['content']):,} chars"
            )

            print(
                f"Links    : "
                f"{document['internal_link_count']}"
            )

    # ========================================================
    # SORT
    # ========================================================

    documents.sort(
        key=lambda item:
        item["url"]
    )

    discovery.sort(
        key=lambda item:
        item["url"]
    )

    # ========================================================
    # SAVE FULL DOCUMENTS
    # ========================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # SAVE DISCOVERY INVENTORY
    # ========================================================

    with open(
        DISCOVERY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            discovery,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    total_chars = sum(
        len(
            item["content"]
        )
        for item in documents
    )

    html_count = sum(
        1
        for item in discovery
        if item["resource_type"]
        == "html"
    )

    pdf_count = sum(
        1
        for item in discovery
        if item["resource_type"]
        == "pdf"
    )

    subdomain_count = sum(
        1
        for item in discovery
        if item["domain_type"]
        == "subdomain"
    )

    external_count = sum(
        1
        for item in discovery
        if item["domain_type"]
        == "external"
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 72)
    print(
        "CRAWL4AI DISCOVERY COMPLETE"
    )
    print("=" * 72)

    print(
        f"Discovered URLs : "
        f"{len(discovery)}"
    )

    print(
        f"HTML documents  : "
        f"{len(documents)}"
    )

    print(
        f"PDF resources   : "
        f"{pdf_count}"
    )

    print(
        f"Subdomains      : "
        f"{subdomain_count}"
    )

    print(
        f"External URLs   : "
        f"{external_count}"
    )

    print(
        f"Characters      : "
        f"{total_chars:,}"
    )

    print(
        f"Documents       : "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Discovery       : "
        f"{DISCOVERY_FILE}"
    )

    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        crawl_fortune_cloud(
            max_depth=5,
            max_pages=200,
        )
    )