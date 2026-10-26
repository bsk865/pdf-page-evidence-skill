# /// script
# requires-python = ">=3.11"
# dependencies = ["pypdf==6.19.0"]
# ///
"""Local physical-page PDF indexing, lexical lookup, and exact quote checks."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import unicodedata

from pypdf import PdfReader


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def citation(digest: str, page: int) -> str:
    return f"pdfsha256:{digest}:page:{page}"


def normalize_quote(text: str) -> str:
    return " ".join(text.split())


def read_pdf(path: Path) -> tuple[str, list[str]]:
    raw = path.read_bytes()
    if not raw.startswith(b"%PDF-"):
        raise ValueError("Input does not start with a PDF header")
    reader = PdfReader(io.BytesIO(raw), strict=True)
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("PDF needs a password; unlock a local copy before indexing")
    pages = [page.extract_text() or "" for page in reader.pages]
    return sha256(raw), pages


def extract(args: argparse.Namespace) -> dict:
    source = args.pdf.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("The index output must differ from the original PDF")
    digest, texts = read_pdf(source)
    rows = [{"type": "document", "schema": 1, "filename": source.name,
             "sha256": digest, "page_count": len(texts)}]
    rows.extend({"type": "page", "page_number": number,
                 "citation": citation(digest, number), "text": text,
                 "text_sha256": sha256(text.encode("utf-8"))}
                for number, text in enumerate(texts, start=1))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=output.parent, prefix=".pdf-index-", delete=False) as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"status": "indexed", "filename": source.name, "sha256": digest,
            "physical_pages": len(texts), "blank_text_pages": sum(not text.strip() for text in texts),
            "output": str(output)}


def load_index(path: Path) -> tuple[dict, list[dict]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or rows[0].get("type") != "document" or rows[0].get("schema") != 1:
        raise ValueError("Unsupported or missing PDF index header")
    header, pages = rows[0], rows[1:]
    digest = header.get("sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or header.get("page_count") != len(pages):
        raise ValueError("Invalid PDF digest or physical-page count")
    for number, page in enumerate(pages, start=1):
        if (page.get("type") != "page" or page.get("page_number") != number
                or page.get("citation") != citation(digest, number) or not isinstance(page.get("text"), str)
                or page.get("text_sha256") != sha256(page["text"].encode("utf-8"))):
            raise ValueError(f"Invalid or changed index page {number}")
    return header, pages


def terms(text: str) -> list[str]:
    # CJK characters remain searchable even when the source has no word spaces.
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[^\W_\u3400-\u9fff]+|[\u3400-\u9fff]", normalized)


def snippet(text: str, query_terms: list[str], width: int = 360) -> str:
    folded = text.casefold()
    positions = [folded.find(term) for term in query_terms if term in folded]
    start = max(0, min(positions, default=0) - 80)
    return text[start:start + width]


def find(args: argparse.Namespace) -> dict:
    if not 1 <= args.limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    header, pages = load_index(args.index)
    query_terms = list(dict.fromkeys(terms(args.query)))
    if not query_terms:
        raise ValueError("Query needs at least one searchable letter, number or CJK character")
    counts = [Counter(terms(page["text"])) for page in pages]
    lengths = [sum(count.values()) for count in counts]
    average = sum(lengths) / max(1, len(pages)) or 1
    frequencies = {term: sum(term in count for count in counts) for term in query_terms}
    ranked = []
    for page, count, length in zip(pages, counts, lengths):
        score = 0.0
        for term in query_terms:
            hits = count[term]
            if hits:
                inverse = math.log(1 + (len(pages) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
                score += inverse * hits * 2.5 / (hits + 1.5 * (0.25 + 0.75 * length / average))
        if score > 0:
            ranked.append((score, page))
    ranked.sort(key=lambda item: (-item[0], item[1]["page_number"]))
    return {"status": "candidate_pages", "filename": header["filename"], "sha256": header["sha256"],
            "query": args.query, "method": "offline lexical BM25; candidates require original-page verification",
            "results": [{"page_number": page["page_number"], "citation": page["citation"],
                         "score": round(score, 6), "excerpt": snippet(page["text"], query_terms)}
                        for score, page in ranked[:args.limit]]}


def verify(args: argparse.Namespace) -> dict:
    match = re.fullmatch(r"pdfsha256:([0-9a-f]{64}):page:([1-9][0-9]*)", args.citation)
    if match is None:
        raise ValueError("Invalid citation; expected pdfsha256:<64 lowercase hex>:page:<physical page>")
    digest, texts = read_pdf(args.pdf)
    if digest != match[1]:
        raise ValueError("Original-file SHA-256 does not match the citation")
    page_number = int(match[2])
    if page_number > len(texts):
        raise ValueError("Physical page is outside this PDF")
    text = texts[page_number - 1]
    if args.quote is not None:
        quote = normalize_quote(args.quote)
        if not quote:
            raise ValueError("Quote must not be empty")
        if quote not in normalize_quote(text):
            raise ValueError("Quote was not found on this physical page")
    return {"status": "verified", "filename": args.pdf.name, "sha256": digest,
            "page_number": page_number, "citation": args.citation,
            "quote_checked": args.quote is not None, "text": text}


def main() -> int:
    # Keep JSON output readable after Windows pipe/file redirection as well as in terminals.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("extract", help="Index every physical page of one local PDF")
    command.add_argument("pdf", type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.set_defaults(function=extract)
    command = commands.add_parser("find", help="Find lexical candidate pages in an index")
    command.add_argument("index", type=Path)
    command.add_argument("--query", required=True)
    command.add_argument("--limit", type=int, default=5)
    command.set_defaults(function=find)
    command = commands.add_parser("verify", help="Check a citation and optional quote against the original PDF")
    command.add_argument("pdf", type=Path)
    command.add_argument("--citation", required=True)
    command.add_argument("--quote")
    command.set_defaults(function=verify)
    args = parser.parse_args()
    try:
        result = args.function(args)
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
