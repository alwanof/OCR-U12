"""Fix reversed Arabic OCR output.

MinerU/PaddleOCR emit recognized Arabic text in visual (left-to-right) order
instead of logical order (opendatalab/MinerU#677). Applying the Unicode bidi
display transform with an RTL base direction converts visual back to logical.

The right place to fix this is span-by-span in MinerU's middle.json (each span
is one physical line), BEFORE markdown assembly — flipping assembled markdown
would also reverse the line order inside merged paragraphs.

The transform is gated by a word-frequency heuristic, so it becomes a no-op
automatically if upstream fixes RTL handling.
"""

import re

from bidi.algorithm import get_display

ARABIC_RE = re.compile(r"[؀-ۿ]")
# Leading markdown syntax (headings, list markers) must not be flipped.
MD_PREFIX_RE = re.compile(r"^(\s*(?:#{1,6}\s+|[-*+]\s+|\d+\.\s+)?)(.*)$", re.S)
HTML_TEXT_RE = re.compile(r">([^<>]+)<")
# Very common Arabic words; matched at word start.
COMMON_TOKENS = ("ال", "في", "من", "على", "إلى", "أن", "إن", "هذا", "هذه", "التي", "الذي")


def fix_middle_json(middle: dict) -> bool:
    """Flip reversed Arabic spans in-place. Returns True if a flip was applied."""
    spans = [s for s in _walk_spans(middle.get("pdf_info", []))]
    texts = [s["content"] for s in spans if _is_arabic(s.get("content", ""))]
    if not _needs_flip("\n".join(texts[:300])):
        return False
    for span in spans:
        content = span.get("content")
        if content and _is_arabic(content):
            span["content"] = _flip(content)
        html = span.get("html")
        if html and _is_arabic(html):
            span["html"] = HTML_TEXT_RE.sub(
                lambda m: ">" + (_flip(m.group(1)) if _is_arabic(m.group(1)) else m.group(1)) + "<",
                html,
            )
    return True


def fix_rtl_markdown(md: str) -> str:
    """Line-level fallback when middle.json is unavailable. May reverse the
    order of merged physical lines inside a paragraph — prefer fix_middle_json."""
    lines = md.splitlines()
    arabic_lines = [l for l in lines if _is_arabic(l) and "](" not in l]
    if not arabic_lines or not _needs_flip("\n".join(arabic_lines[:200])):
        return md
    fixed = []
    for line in lines:
        if not _is_arabic(line) or "](" in line:
            fixed.append(line)
        elif "|" in line:  # markdown table row: flip cell contents, keep structure
            fixed.append("|".join(_flip(cell) for cell in line.split("|")))
        else:
            m = MD_PREFIX_RE.match(line)
            fixed.append(m.group(1) + _flip(m.group(2)))
    return "\n".join(fixed) + "\n"


def _walk_spans(node):
    if isinstance(node, dict):
        for span in node.get("spans") or []:
            if isinstance(span, dict):
                yield span
        for key in ("blocks", "lines", "para_blocks"):
            yield from _walk_spans(node.get(key))
    elif isinstance(node, list):
        for item in node:
            yield from _walk_spans(item)


def _is_arabic(text: str) -> bool:
    return bool(ARABIC_RE.search(text))


def _needs_flip(sample: str) -> bool:
    if not sample:
        return False
    flipped = "\n".join(_flip(l) for l in sample.splitlines())
    return _score(flipped) > _score(sample)


def _flip(text: str) -> str:
    return get_display(text, base_dir="R")


def _score(text: str) -> int:
    return sum(len(re.findall(r"(?<![؀-ۿ])" + tok, text)) for tok in COMMON_TOKENS)
