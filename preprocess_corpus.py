#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import struct
from pathlib import Path


DOT_LEADER_RE = re.compile(r"(?:\.\s*){4,}")
COPYRIGHT_RE = re.compile(
    r"(copyright|licensed under|creative commons|all rights reserved)",
    re.IGNORECASE,
)
LAYOUT_EXPORT_RE = re.compile(r".*\.(?:qxd|pdf|indd)$", re.IGNORECASE)
DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?:\s*(?:AM|PM|am|pm))?\b")
PAGE_MARKER_RE = re.compile(r"^Page\s+\d+\b", re.IGNORECASE)
BULLET_PREFIX_RE = re.compile(r"^[◆•▪►–—*]+\s*")
TERMINAL_PUNCT_RE = re.compile(r"[.?!][\"')\]]*$")
INLINE_CITATION_RE = re.compile(
    r"\[(?:RFC\s*\d+|Page\s*\d+|[A-Z0-9_-]+)\]",
    re.IGNORECASE,
)
HTTP_URL_RE = re.compile(r"<?https?://[^>\s]+>?", re.IGNORECASE)
DOI_RE = re.compile(r"\bDOI\s+10\.\d{4,9}/\S+", re.IGNORECASE)
MULTISPACE_RE = re.compile(r"\s{2,}")
RFC_HEADER_FOOTER_RE = re.compile(
    r"(?:George\s*&\s*Amante|Standards\s+Track|\[Page\s+\d+\]|\bRFC\s+\d+\b)",
    re.IGNORECASE,
)
MONTH_YEAR_RE = re.compile(
    r"^(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}$",
    re.IGNORECASE,
)
FIGURE_CAPTION_RE = re.compile(r"^Figure\s+\d+[A-Za-z]?(?::|\.)\s+", re.IGNORECASE)
SECTION_TITLE_RE = re.compile(r"^\d+(?:\.\d+)+\.?\s+\S")
REFERENCE_HEADER_RE = re.compile(
    r"^\d+(?:\.\d+)*\.\s+(?:Normative|Informative)\s+References$",
    re.IGNORECASE,
)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
META_STRUCT = struct.Struct("<HIQH")
MAX_TEXT_BYTES = 65535
MAX_FILE_IDS = 65536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract, clean, and serialize a text corpus for search indexing."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/mnt/c/Users/shire/Desktop/Archive"),
        help="Root directory containing PDF-converted .txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/c/Users/shire/Desktop/processed_data"),
        help="Directory where processed corpus artifacts will be written.",
    )
    return parser.parse_args()


def iter_text_files(root: Path):
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        for filename in filenames:
            if filename.startswith("."):
                continue
            if not filename.lower().endswith(".txt"):
                continue
            yield Path(current_root) / filename


def read_text_lines(file_path: Path):
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            yield from handle
        return
    except UnicodeDecodeError:
        pass

    with file_path.open("r", encoding="latin-1", errors="replace") as handle:
        yield from handle


def trim_utf8_bytes(text: str, max_bytes: int = MAX_TEXT_BYTES) -> bytes:
    encoded = text.encode("utf-8", errors="strict")
    if len(encoded) <= max_bytes:
        return encoded

    trimmed = text
    while trimmed:
        overflow_ratio = len(encoded) / max_bytes
        next_length = max(1, int(len(trimmed) / max(overflow_ratio, 1.1)))
        trimmed = trimmed[:next_length]
        encoded = trimmed.encode("utf-8", errors="strict")
        if len(encoded) <= max_bytes:
            return encoded

    return b""


def is_dot_leader_line(text: str) -> bool:
    if DOT_LEADER_RE.search(text):
        return True

    dot_count = text.count(".")
    return bool(text) and (dot_count / len(text)) > 0.25


def clean_line(raw_line: str) -> str:
    cleaned = CONTROL_CHAR_RE.sub("", raw_line).strip(" \t\r\n")
    if not cleaned:
        return ""

    cleaned = BULLET_PREFIX_RE.sub("", cleaned).strip(" \t")
    cleaned = INLINE_CITATION_RE.sub("", cleaned)
    cleaned = HTTP_URL_RE.sub("", cleaned)
    cleaned = DOI_RE.sub("", cleaned)
    cleaned = MULTISPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def normalize_line(raw_line: str) -> str:
    # Backward-compatible alias for call sites that may still use normalize_line.
    return clean_line(raw_line)


def should_skip_line(text: str) -> tuple[bool, bool]:
    """
    Return (skip_line, section_break).
    section_break=True means the current sentence buffer should be flushed.
    """
    if len(text) < 3:
        return True, False
    if LAYOUT_EXPORT_RE.fullmatch(text):
        return True, True
    if DATE_RE.search(text):
        return True, True
    if TIME_RE.search(text):
        return True, True
    if PAGE_MARKER_RE.match(text):
        return True, True
    if text.isdigit():
        return True, True
    if is_dot_leader_line(text):
        return True, False
    if COPYRIGHT_RE.search(text):
        return True, False
    if RFC_HEADER_FOOTER_RE.search(text):
        return True, True
    if MONTH_YEAR_RE.fullmatch(text):
        return True, True
    return False, False


def ends_sentence(text: str) -> bool:
    if text.endswith((".", "?", "!")):
        return True
    return TERMINAL_PUNCT_RE.search(text) is not None


def is_valid_sentence(text: str) -> bool:
    if not text:
        return False
    if RFC_HEADER_FOOTER_RE.search(text):
        return False
    if MONTH_YEAR_RE.fullmatch(text):
        return False
    if REFERENCE_HEADER_RE.fullmatch(text):
        return False
    if FIGURE_CAPTION_RE.match(text) and not ends_sentence(text):
        return False
    if SECTION_TITLE_RE.match(text) and not ends_sentence(text):
        return False
    return True


def process_corpus(input_dir: Path, output_dir: Path) -> None:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_corpus_path = output_dir / "raw_corpus.bin"
    meta_path = output_dir / "sentences_meta.bin"
    files_map_path = output_dir / "files_map.json"

    text_files = sorted(iter_text_files(input_dir))
    if len(text_files) > MAX_FILE_IDS:
        raise ValueError(
            f"Found {len(text_files)} text files, but metadata supports at most {MAX_FILE_IDS} files"
        )

    files_map: dict[str, str] = {}
    total_sentences = 0
    char_offset = 0

    with raw_corpus_path.open("wb") as raw_handle, meta_path.open("wb") as meta_handle:
        for file_id, file_path in enumerate(text_files):
            files_map[str(file_id)] = str(file_path)
            previous_line: str | None = None
            sentence_parts: list[str] = []
            sentence_start_line: int | None = None

            def flush_buffer() -> None:
                nonlocal total_sentences, char_offset, sentence_start_line, sentence_parts
                if not sentence_parts:
                    return

                sentence = " ".join(sentence_parts).strip()
                start_line = sentence_start_line
                sentence_parts = []
                sentence_start_line = None

                if len(sentence) < 15 or len(sentence.split()) < 3:
                    return

                encoded = trim_utf8_bytes(sentence)
                if not encoded:
                    return

                text_length = len(encoded)
                raw_handle.write(encoded)
                raw_handle.write(b"\n")
                meta_handle.write(
                    META_STRUCT.pack(file_id, start_line or 1, char_offset, text_length)
                )

                char_offset += text_length + 1
                total_sentences += 1

            for line_number, raw_line in enumerate(read_text_lines(file_path), start=1):
                cleaned = clean_line(raw_line)

                if not cleaned:
                    flush_buffer()
                    previous_line = None
                    continue

                skip_line, section_break = should_skip_line(cleaned)
                if skip_line:
                    if section_break:
                        flush_buffer()
                        previous_line = None
                    continue

                if cleaned == previous_line:
                    continue
                if not is_valid_sentence(cleaned):
                    flush_buffer()
                    previous_line = None
                    continue

                previous_line = cleaned
                if sentence_start_line is None:
                    sentence_start_line = line_number

                sentence_parts.append(cleaned)
                if ends_sentence(cleaned):
                    flush_buffer()

            flush_buffer()

    with files_map_path.open("w", encoding="utf-8") as map_handle:
        json.dump(files_map, map_handle, ensure_ascii=False, indent=2)

    raw_size = raw_corpus_path.stat().st_size
    meta_size = meta_path.stat().st_size
    map_size = files_map_path.stat().st_size

    assert meta_size == total_sentences * META_STRUCT.size, (
        f"sentences_meta.bin size mismatch: expected {total_sentences * META_STRUCT.size} "
        f"bytes, got {meta_size} bytes"
    )

    print(f"Total raw files processed: {len(text_files)}")
    print(f"Total clean sentences created: {total_sentences}")
    print(f"raw_corpus.bin size: {raw_size / (1024 * 1024):.2f} MB")
    print(f"sentences_meta.bin size: {meta_size / (1024 * 1024):.2f} MB")
    print(f"files_map.json size: {map_size / (1024 * 1024):.2f} MB")


def main() -> None:
    args = parse_args()
    process_corpus(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()