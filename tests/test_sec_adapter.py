"""
Tests for the SEC HTML cleaning/section-splitting logic in
src/ingest/sources/sec.py.

TEST FIXTURE -- NOT PROJECT DATA. The HTML below is a small hand-written
fixture, not a real filing, built specifically to pin down two real bugs
found while inspecting the actual downloaded corpus:

    1. Some filers render "ITEM 1." in all caps; an earlier, case-sensitive
       version of the header regex silently missed these (confirmed against
       AMD's real filing).
    2. Extracting text with get_text(separator="\\n") breaks a heading or
       sentence apart wherever the filer used multiple inline <span> tags
       for styling (confirmed against MSFT's real filing, which produced
       "ITEM 1A. RIS" + a disconnected "K FACTORS" fragment). Extracting
       one paragraph per leaf block-level element instead fixes this.
"""

from src.ingest.sources.sec import _extract_paragraphs, _split_into_sections
from bs4 import BeautifulSoup

_FIXTURE_HTML = """
<html><body>
<div style="display:none"><ix:header>should never appear in extracted text</ix:header></div>
<div>Table of Contents</div>
<div>Item 1.</div>
<div>Item 1A.</div>
<div>PART I</div>
<div><span style="color:red">ITEM 1.</span><span style="color:black"> BUSINESS</span></div>
<div>This company designs and sells widgets. """ + ("Widgets are useful. " * 40) + """</div>
<div>ITEM 1A.</div>
<div>Risk factors include competition and supply chain risk. """ + ("Risk detail. " * 40) + """</div>
</body></html>
"""


def test_hidden_xbrl_block_is_excluded_from_paragraphs():
    soup = BeautifulSoup(_FIXTURE_HTML, "lxml")
    for tag in soup.find_all(style=True):
        if "display:none" in tag.get("style", "").replace(" ", "").lower():
            tag.decompose()
    paragraphs = _extract_paragraphs(soup)
    assert not any("should never appear" in p for p in paragraphs)


def test_inline_span_split_heading_is_reconstructed_whole():
    soup = BeautifulSoup(_FIXTURE_HTML, "lxml")
    for tag in soup.find_all(style=True):
        if "display:none" in tag.get("style", "").replace(" ", "").lower():
            tag.decompose()
    paragraphs = _extract_paragraphs(soup)
    assert "ITEM 1. BUSINESS" in paragraphs


def test_toc_mention_loses_to_real_heading_with_more_body_text():
    soup = BeautifulSoup(_FIXTURE_HTML, "lxml")
    for tag in soup.find_all(style=True):
        if "display:none" in tag.get("style", "").replace(" ", "").lower():
            tag.decompose()
    paragraphs = _extract_paragraphs(soup)
    sections = _split_into_sections(paragraphs)

    by_path = {s.section_path: s for s in sections}
    assert "ITEM 1. BUSINESS" in by_path
    assert "designs and sells widgets" in by_path["ITEM 1. BUSINESS"].text
    assert "ITEM 1A." in by_path
    assert "Risk factors include competition" in by_path["ITEM 1A."].text
    # the bare TOC mentions ("Item 1.", "Item 1A." -- title case, no body after)
    # must lose to the real headings above and never become their own section
    assert "Item 1." not in by_path
    assert "Item 1A." not in by_path
