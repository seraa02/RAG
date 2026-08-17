"""
SEC EDGAR source adapter.

Purpose:
    This is the ONLY module in the system allowed to know that a document
    came from SEC EDGAR / is a 10-K / is HTML. It owns everything
    source-specific: downloading, SEC HTML structure parsing, and
    deterministic structural cleaning (stripping nav menus, headers,
    footers, page numbers -- never rewriting wording).

    Its output must be a NormalizedDocument (see normalize.py). Nothing
    downstream of this module is allowed to know or care that the source
    was SEC EDGAR -- that is the architectural boundary described in
    README.md.

Status:
    Download step is implemented and idempotent -- re-running
    download_corpus() skips any ticker already present in manifest.json
    with its file still on disk.

    Cleaning/normalization (clean_and_normalize / normalize_corpus) is
    implemented after inspecting the real downloaded HTML: these filings
    are Workiva-generated inline-XBRL documents -- no semantic HTML5, a
    hidden `display:none` block carrying XBRL financial tags that must
    be stripped, and a Table of Contents whose entries are textually
    near-identical to the real section headings they link to. Section
    boundaries (Item 1, Item 1A, ...) are detected by finding, for each
    Item number, the occurrence followed by the most body text before
    the next heading -- the TOC occurrence is followed almost immediately
    by the next TOC entry, while the real heading is followed by the
    actual section. This is a heuristic, not a guarantee, on every
    filer's HTML.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

import requests

from src.config import get_settings
from src.ingest.normalize import build_normalized_document
from src.schema import DocumentSection, FilingMetadata, NormalizedDocument

TARGET_TICKERS: list[str] = ["NVDA", "AMD", "INTC", "MSFT", "AMZN", "GOOGL"]

_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_ARCHIVE_URL_TEMPLATE = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{primary_document}"
)

_REQUEST_DELAY_SECONDS = 0.3  # well under SEC's 10 req/sec fair-access limit

_RAW_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "raw"
_MANIFEST_PATH = _RAW_DIR / "manifest.json"


def _session() -> requests.Session:
    settings = get_settings()
    s = requests.Session()
    s.headers.update(
        {"User-Agent": settings.sec_user_agent, "Accept-Encoding": "gzip, deflate"}
    )
    return s


def resolve_tickers(
    tickers: list[str], session: requests.Session
) -> dict[str, dict[str, str]]:
    """
    Resolve tickers to zero-padded 10-digit CIKs (and official company
    titles) using SEC's canonical ticker->CIK mapping file, rather than
    hardcoding CIK numbers in source.
    """
    resp = session.get(_COMPANY_TICKERS_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()  # {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    by_ticker = {entry["ticker"]: entry for entry in data.values()}

    result: dict[str, dict[str, str]] = {}
    missing = []
    for ticker in tickers:
        entry = by_ticker.get(ticker)
        if entry is None:
            missing.append(ticker)
            continue
        result[ticker] = {"cik": f"{entry['cik_str']:010d}", "title": entry["title"]}
    if missing:
        raise ValueError(f"Could not resolve CIK for tickers: {missing}")
    return result


def get_latest_10k(cik: str, session: requests.Session) -> dict:
    """Return the most recent 10-K filing record for a company, from its
    SEC submissions feed (form, filingDate, reportDate, accessionNumber,
    primaryDocument)."""
    url = _SUBMISSIONS_URL_TEMPLATE.format(cik=int(cik))
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    recent = data["filings"]["recent"]
    candidates = [
        {
            "form": recent["form"][i],
            "filingDate": recent["filingDate"][i],
            "reportDate": recent["reportDate"][i],
            "accessionNumber": recent["accessionNumber"][i],
            "primaryDocument": recent["primaryDocument"][i],
        }
        for i in range(len(recent["form"]))
        if recent["form"][i] == "10-K"
    ]
    if not candidates:
        raise ValueError(f"No 10-K filing found for CIK {cik}")
    return max(candidates, key=lambda f: f["filingDate"])


def download_filing_html(cik: str, filing: dict, session: requests.Session) -> bytes:
    accession_nodash = filing["accessionNumber"].replace("-", "")
    url = _ARCHIVE_URL_TEMPLATE.format(
        cik=int(cik),
        accession_nodash=accession_nodash,
        primary_document=filing["primaryDocument"],
    )
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def _load_manifest() -> dict[str, dict]:
    if _MANIFEST_PATH.exists():
        records = json.loads(_MANIFEST_PATH.read_text())
        return {r["ticker"]: r for r in records}
    return {}


def _save_manifest(manifest_by_ticker: dict[str, dict]) -> None:
    _MANIFEST_PATH.write_text(json.dumps(list(manifest_by_ticker.values()), indent=2))


def download_corpus(tickers: list[str] = TARGET_TICKERS) -> list[FilingMetadata]:
    """
    Download the most recent 10-K for each ticker into data/raw/,
    recording each download in data/raw/manifest.json. Idempotent: a
    ticker already present in the manifest whose file still exists on
    disk is skipped, not re-downloaded (Rule #12).
    """
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = _session()
    manifest_by_ticker = _load_manifest()

    to_resolve = [t for t in tickers if t not in manifest_by_ticker or not Path(
        manifest_by_ticker[t]["local_path"]
    ).exists()]

    resolved: dict[str, dict[str, str]] = {}
    if to_resolve:
        print(f"Resolving CIKs for {to_resolve}...")
        resolved = resolve_tickers(to_resolve, session)
        time.sleep(_REQUEST_DELAY_SECONDS)

    results: list[FilingMetadata] = []
    for ticker in tickers:
        existing = manifest_by_ticker.get(ticker)
        if existing and Path(existing["local_path"]).exists():
            print(f"[skip] {ticker}: already downloaded -> {existing['local_path']}")
            results.append(FilingMetadata(**existing))
            continue

        info = resolved[ticker]
        cik, company_name = info["cik"], info["title"]

        print(f"[fetch] {ticker} (CIK {cik}): looking up latest 10-K...")
        filing = get_latest_10k(cik, session)
        time.sleep(_REQUEST_DELAY_SECONDS)

        print(
            f"[fetch] {ticker}: downloading {filing['primaryDocument']} "
            f"(filed {filing['filingDate']})..."
        )
        content = download_filing_html(cik, filing, session)
        time.sleep(_REQUEST_DELAY_SECONDS)

        fiscal_year = filing["reportDate"][:4]
        local_path = _RAW_DIR / f"{ticker}_{fiscal_year}_10K.html"
        local_path.write_bytes(content)

        record = FilingMetadata(
            company=company_name,
            ticker=ticker,
            cik=cik,
            accession_number=filing["accessionNumber"],
            filing_type="10-K",
            filing_date=filing["filingDate"],
            local_path=str(local_path),
            downloaded_at=datetime.now(timezone.utc).isoformat(),
            sha256=hashlib.sha256(content).hexdigest(),
        )
        results.append(record)
        manifest_by_ticker[ticker] = record.model_dump(mode="json")
        print(f"[done] {ticker}: {len(content):,} bytes -> {local_path}")

    _save_manifest(manifest_by_ticker)
    return results


_PROCESSED_DIR = _RAW_DIR.parent / "processed"

# A candidate section header: "PART I", "Item 1.", "Item 1A.", optionally
# followed by the section title on the same line (e.g. "Item 1A. Risk Factors").
# Case-insensitive: some filers render "Item 1." (title case, title on the
# same line), others "ITEM 1." (all caps, title on the following line) --
# verified by inspecting AMD's raw HTML, which uses the all-caps form and
# was silently missed by an earlier case-sensitive version of this regex.
_ITEM_HEADER_RE = re.compile(r"^(PART\s+[IVX]+|Item\s+\d+[A-Za-z]?\.)\s*(.*)$", re.IGNORECASE)
_MAX_HEADER_LINE_LENGTH = 100  # longer than this, it's body prose that happens to match, not a heading
_NAV_NOISE = {"table of contents"}  # recurring nav link text, zero information value


def _strip_hidden_and_noise(soup: BeautifulSoup) -> None:
    """Remove <script>/<style> and any element inline-styled display:none --
    the latter is where inline-XBRL financial tagging metadata lives in these
    filings. None of this is visible text a reader (or citation) should ever
    surface."""
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(style=True):
        if "display:none" in tag.get("style", "").replace(" ", "").lower():
            tag.decompose()


_BLOCK_TAGS = ("div", "p", "td", "li", "h1", "h2", "h3", "h4", "h5", "h6")


def _iter_leaf_blocks(soup: BeautifulSoup):
    """Yield block-level elements with no nested block-level descendant --
    i.e. the innermost container actually holding a paragraph's text. Using
    get_text(separator="\\n") on the whole soup instead (tried first) inserts
    a break between *every* tag boundary, including inline <span> runs used
    for mid-word color/weight changes -- verified against MSFT's raw HTML,
    where it split headings like "Item 1A. Risk Factors" into "ITEM 1A. RIS"
    + a disconnected "K FACTORS" fragment. Walking leaf blocks keeps each
    real paragraph/heading whole regardless of inline span nesting."""
    for tag in soup.find_all(_BLOCK_TAGS):
        if not tag.find(_BLOCK_TAGS):
            yield tag


def _extract_paragraphs(soup: BeautifulSoup) -> list[str]:
    """Flatten cleaned HTML into a list of non-empty paragraph strings, one
    per leaf block-level element. Whitespace is collapsed; wording is
    untouched. Drops two confirmed noise patterns -- recurring "Table of
    Contents" nav text, and bare page-number-like lines (<=5 digits alone)
    -- both explicitly named as noise to strip in README's cleaning
    philosophy."""
    paragraphs = []
    for block in _iter_leaf_blocks(soup):
        text = re.sub(r"\s+", " ", block.get_text(separator=" ")).strip()
        if not text:
            continue
        if text.lower() in _NAV_NOISE:
            continue
        if text.isdigit() and len(text) <= 5:
            continue
        paragraphs.append(text)
    return paragraphs


def _split_into_sections(paragraphs: list[str]) -> list[DocumentSection]:
    """Group paragraphs into DocumentSections keyed by detected Item/PART
    headings, picking the real body heading over Table-of-Contents mentions
    of the same heading (see module docstring)."""
    candidates: list[tuple[int, str, str]] = []  # (paragraph_index, key, title)
    for i, para in enumerate(paragraphs):
        match = _ITEM_HEADER_RE.match(para)
        if match and len(para) <= _MAX_HEADER_LINE_LENGTH:
            key = match.group(1).rstrip(".").upper().replace(" ", "_")
            candidates.append((i, key, para))

    if not candidates:
        return [DocumentSection(section_path="Full Document", text="\n\n".join(paragraphs))]

    candidate_indices = {c[0] for c in candidates}

    def body_length_after(idx: int) -> int:
        end = len(paragraphs)
        for j in range(idx + 1, len(paragraphs)):
            if j in candidate_indices:
                end = j
                break
        return sum(len(p) for p in paragraphs[idx + 1 : end])

    best_for_key: dict[str, tuple[int, str, int]] = {}
    for idx, key, title in candidates:
        length = body_length_after(idx)
        if key not in best_for_key or length > best_for_key[key][2]:
            best_for_key[key] = (idx, title, length)

    chosen = sorted(best_for_key.values(), key=lambda t: t[0])

    sections: list[DocumentSection] = []
    if chosen[0][0] > 0:
        sections.append(
            DocumentSection(section_path="Front Matter", text="\n\n".join(paragraphs[: chosen[0][0]]))
        )
    for n, (idx, title, _length) in enumerate(chosen):
        end = chosen[n + 1][0] if n + 1 < len(chosen) else len(paragraphs)
        sections.append(
            DocumentSection(section_path=title, text="\n\n".join(paragraphs[idx + 1 : end]))
        )
    return sections


def clean_and_normalize(record: FilingMetadata) -> NormalizedDocument:
    """Parse the raw downloaded HTML for one filing, strip structural noise,
    detect section boundaries, and emit a NormalizedDocument. data/raw/ is
    only ever read here, never modified."""
    raw_bytes = Path(record.local_path).read_bytes()
    soup = BeautifulSoup(raw_bytes, "lxml")
    _strip_hidden_and_noise(soup)

    paragraphs = _extract_paragraphs(soup)
    sections = _split_into_sections(paragraphs)
    full_text = "\n\n".join(paragraphs)

    document_id = Path(record.local_path).stem  # e.g. "NVDA_2026_10K"
    fiscal_year = document_id.split("_")[1]

    return build_normalized_document(
        document_id=document_id,
        source="sec_edgar",
        document_type=record.filing_type,
        title=f"{record.company} {record.filing_type} (FY{fiscal_year})",
        date=record.filing_date,
        metadata={
            "company": record.company,
            "ticker": record.ticker,
            "cik": record.cik,
            "accession_number": record.accession_number,
            "fiscal_year": fiscal_year,
        },
        sections=sections,
        text=full_text,
        source_uri=(
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(record.cik)}/{record.accession_number.replace('-', '')}/"
        ),
        content_hash=record.sha256,
    )


def normalize_corpus(records: list[FilingMetadata] | None = None) -> list[NormalizedDocument]:
    """Clean + normalize every downloaded filing, writing one JSON file per
    document into data/processed/. Idempotent by construction: always
    re-derived from data/raw/, which is never modified."""
    if records is None:
        records = [FilingMetadata(**r) for r in _load_manifest().values()]

    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    documents = []
    for record in records:
        doc = clean_and_normalize(record)
        out_path = _PROCESSED_DIR / f"{doc.document_id}.json"
        out_path.write_text(doc.model_dump_json(indent=2))
        documents.append(doc)
        print(f"[normalized] {doc.document_id}: {len(doc.sections)} sections, {len(doc.text):,} chars")
    return documents


if __name__ == "__main__":
    downloaded = download_corpus()
    print(f"\n{len(downloaded)} filings available in {_RAW_DIR}")
    normalized = normalize_corpus(downloaded)
    print(f"{len(normalized)} documents normalized into {_PROCESSED_DIR}")
