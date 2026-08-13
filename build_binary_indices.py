#!/usr/bin/env python3

import argparse
import json
import os
import re
import struct


SENTENCE_SPLIT_RE = re.compile(r"(?<=\w\.)\s+")
NORMALIZE_RE = re.compile(r"[^a-z0-9 ]+")
MULTISPACE_RE = re.compile(r"\s+")

META_STRUCT = struct.Struct("<HIQH")
BIGRAM_DICT_STRUCT = struct.Struct("<2s2xIQ")
TRIGRAM_DICT_STRUCT = struct.Struct("<3s1xIQ")

MAX_UINT16 = 65535
MAX_UINT32 = 4294967295


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process text corpus and build binary sentence and n-gram indices."
    )
    parser.add_argument(
        "--input-dir",
        default="raw_text_files",
        help="Directory containing source .txt files (recursively scanned).",
    )
    parser.add_argument(
        "--output-dir",
        default="processed_data",
        help="Directory to write all generated output files.",
    )
    return parser.parse_args()


def reset_output_dir(output_dir):
    if os.path.isdir(output_dir):
        for root, dirs, files in os.walk(output_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    os.makedirs(output_dir, exist_ok=True)


def iter_text_files(input_dir):
    for root, dirs, files in os.walk(input_dir):
        dirs.sort()
        files.sort()
        for name in files:
            if name.lower().endswith(".txt"):
                yield os.path.join(root, name)


def normalize_for_ngrams(sentence):
    text = sentence.lower()
    text = NORMALIZE_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def trim_utf8_to_uint16(text):
    encoded = text.encode("utf-8")
    if len(encoded) <= MAX_UINT16:
        return encoded

    # Trim safely by Unicode code points until encoded byte length fits uint16.
    while text:
        text = text[:-1]
        encoded = text.encode("utf-8")
        if len(encoded) <= MAX_UINT16:
            return encoded
    return b""


def split_line_to_sentences(line_text):
    parts = SENTENCE_SPLIT_RE.split(line_text)
    for part in parts:
        sentence = part.strip()
        if sentence:
            yield sentence


def build_sentence_files(input_dir, output_dir):
    raw_corpus_path = os.path.join(output_dir, "raw_corpus.bin")
    files_map_path = os.path.join(output_dir, "files_map.json")
    meta_path = os.path.join(output_dir, "sentences_meta.bin")

    files_map = {}
    sentence_count = 0
    char_offset = 0

    with open(raw_corpus_path, "wb") as raw_handle, open(meta_path, "wb") as meta_handle:
        for file_id, file_path in enumerate(iter_text_files(input_dir)):
            if file_id > MAX_UINT16:
                raise ValueError("file_id exceeds uint16 range")

            files_map[str(file_id)] = file_path
            source_sentence_index = 0

            with open(file_path, "r", encoding="utf-8", errors="replace") as source:
                for raw_line in source:
                    # Newline is always a hard split boundary.
                    line_text = raw_line.rstrip("\r\n")
                    if not line_text.strip():
                        continue

                    # Additional split boundary for "word. <spaces>" within the line.
                    for sentence in split_line_to_sentences(line_text):
                        encoded = trim_utf8_to_uint16(sentence)
                        if not encoded:
                            continue

                        source_sentence_index += 1
                        if source_sentence_index > MAX_UINT32:
                            raise ValueError(
                                f"line_number exceeds uint32 range in file: {file_path}"
                            )

                        text_length = len(encoded)
                        raw_handle.write(encoded)
                        raw_handle.write(b"\n")

                        meta_handle.write(
                            META_STRUCT.pack(
                                file_id,
                                source_sentence_index,
                                char_offset,
                                text_length,
                            )
                        )

                        char_offset += text_length + 1
                        sentence_count += 1

    with open(files_map_path, "w", encoding="utf-8") as map_handle:
        json.dump(files_map, map_handle, ensure_ascii=False, indent=2)

    return {
        "raw_corpus_path": raw_corpus_path,
        "files_map_path": files_map_path,
        "meta_path": meta_path,
        "sentence_count": sentence_count,
        "file_count": len(files_map),
    }


def build_ngram_postings(raw_corpus_path):
    bigram_index = {}
    trigram_index = {}

    with open(raw_corpus_path, "r", encoding="utf-8") as corpus:
        for sentence_id, line in enumerate(corpus):
            sentence = line.rstrip("\n")
            normalized = normalize_for_ngrams(sentence)
            if not normalized:
                continue

            bigrams = set()
            trigrams = set()

            for i in range(len(normalized) - 1):
                bigrams.add(normalized[i : i + 2])

            for i in range(len(normalized) - 2):
                trigrams.add(normalized[i : i + 3])

            for bg in bigrams:
                if bg not in bigram_index:
                    bigram_index[bg] = []
                bigram_index[bg].append(sentence_id)

            for tg in trigrams:
                if tg not in trigram_index:
                    trigram_index[tg] = []
                trigram_index[tg].append(sentence_id)

    return bigram_index, trigram_index


def write_ngram_files(index_data, dict_path, postings_path, dict_struct, n):
    posting_offset = 0

    with open(dict_path, "wb") as dict_handle, open(postings_path, "wb") as post_handle:
        for key in sorted(index_data.keys()):
            postings = index_data[key]
            postings.sort()
            posting_count = len(postings)

            if posting_count > MAX_UINT32:
                raise ValueError(f"posting_count exceeds uint32 range for key: {key}")

            key_bytes = key.encode("ascii", errors="strict")
            if len(key_bytes) != n:
                continue

            for sentence_id in postings:
                if sentence_id < 0 or sentence_id > MAX_UINT32:
                    raise ValueError(f"sentence_id exceeds uint32 range: {sentence_id}")

            if posting_count:
                post_handle.write(struct.pack(f"<{posting_count}I", *postings))

            dict_handle.write(dict_struct.pack(key_bytes, posting_count, posting_offset))
            posting_offset += posting_count * 4


def build_ngram_files(output_dir, raw_corpus_path):
    bigram_dict_path = os.path.join(output_dir, "bigram_dict.bin")
    bigram_postings_path = os.path.join(output_dir, "bigram_postings.bin")
    trigram_dict_path = os.path.join(output_dir, "trigram_dict.bin")
    trigram_postings_path = os.path.join(output_dir, "trigram_postings.bin")

    bigram_index, trigram_index = build_ngram_postings(raw_corpus_path)

    write_ngram_files(
        index_data=bigram_index,
        dict_path=bigram_dict_path,
        postings_path=bigram_postings_path,
        dict_struct=BIGRAM_DICT_STRUCT,
        n=2,
    )

    write_ngram_files(
        index_data=trigram_index,
        dict_path=trigram_dict_path,
        postings_path=trigram_postings_path,
        dict_struct=TRIGRAM_DICT_STRUCT,
        n=3,
    )

    return {
        "bigram_dict_path": bigram_dict_path,
        "bigram_postings_path": bigram_postings_path,
        "trigram_dict_path": trigram_dict_path,
        "trigram_postings_path": trigram_postings_path,
        "bigram_count": len(bigram_index),
        "trigram_count": len(trigram_index),
    }


def validate_meta_file(meta_path, sentence_count):
    expected_size = sentence_count * META_STRUCT.size
    actual_size = os.path.getsize(meta_path)
    if expected_size != actual_size:
        raise RuntimeError(
            f"sentences_meta.bin size mismatch: expected {expected_size}, got {actual_size}"
        )


def main():
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    reset_output_dir(args.output_dir)

    sentence_info = build_sentence_files(args.input_dir, args.output_dir)
    validate_meta_file(sentence_info["meta_path"], sentence_info["sentence_count"])

    ngram_info = build_ngram_files(args.output_dir, sentence_info["raw_corpus_path"])

    print(f"Processed sentences: {sentence_info['sentence_count']}")
    print(f"Files mapped: {sentence_info['file_count']}")
    print(f"Unique bigrams: {ngram_info['bigram_count']}")
    print(f"Unique trigrams: {ngram_info['trigram_count']}")
    print(f"Output written to: {args.output_dir}")


if __name__ == "__main__":
    main()
