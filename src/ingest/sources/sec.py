"""
SEC EDGAR source adapter.

Purpose:
    This is the ONLY module in the system allowed to know that a document
    came from SEC EDGAR / is a 10-K / is HTML. It owns everything
    source-specific: downloading, SEC HTML structure parsing, and
    deterministic structural cleaning (stripping nav menus, headers,
    footers, page numbers -- never rewriting wording).

    Its output must be a NormalizedDocument (see normalize.py). Nothing
    downstream of this module (chunk.py, embed.py, extract.py, resolve.py,
    graph_writer.py) is allowed to know or care that the source was SEC
    EDGAR -- that is the architectural boundary described in README.md
    ("source-specific logic ends at the normalization boundary").

    Adding a second source later (e.g. src/ingest/sources/pdf.py) means
    writing another adapter that emits the same NormalizedDocument shape --
    the shared pipeline does not change.

TODO (Phase 0):
    - Resolve ticker -> CIK for the six target companies.
    - Send a compliant SEC_USER_AGENT header on every request.
    - Find the most recent 10-K per company; download raw HTML to
      data/raw/, unmodified.
    - Compute SHA-256 of each downloaded file; record in manifest.json
      (company, ticker, CIK, accession number, filing type, filing date,
      local path, download timestamp, SHA-256). Skip re-downloading a
      filing already present in the manifest (idempotent -- Rule #12).
    - Parse SEC HTML structure (identify Items/sections) with
      BeautifulSoup.
    - Deterministically strip structural noise (nav, headers, footers,
      page numbers, excessive whitespace) WITHOUT altering wording.
    - Emit one NormalizedDocument per filing.
"""
