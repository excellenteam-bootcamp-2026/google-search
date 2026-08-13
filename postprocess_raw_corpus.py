#!/usr/bin/env python3

import json
import os
import re
import struct


SPLIT_RE = re.compile(r"(?<=\w\.)\s+|\n+")
WHITESPACE_RE = re.compile(r"\s+")
META_STRUCT = struct.Struct("<HIQH")

MAX_UINT16 = 65535
MAX_UINT32 = 4294967295


def trim_utf8_to_uint16(text):
    encoded = text.encode("utf-8")
    if len(encoded) <= MAX_UINT16:
        return encoded

    while text:
        text = text[:-1]
        encoded = text.encode("utf-8")
        if len(encoded) <= MAX_UINT16:
            return encoded

    return b""


def split_to_sentences(raw_text):
    for part in SPLIT_RE.split(raw_text):
        sentence = WHITESPACE_RE.sub(" ", part).strip()
        if sentence:
            yield sentence


def load_and_normalize_files_map(files_map_path):
    if not os.path.isfile(files_map_path):
        raise FileNotFoundError(f"Missing files_map.json: {files_map_path}")

    with open(files_map_path, "r", encoding="utf-8") as handle:
        raw_map = json.load(handle)

    normalized = {}
    for key, value in raw_map.items():
        file_id = int(str(key))
        if file_id < 0 or file_id > MAX_UINT16:
            raise ValueError(f"file_id out of uint16 range in files_map.json: {file_id}")
        if not value:
            raise ValueError(f"files_map.json contains an empty path for file_id {file_id}")
        normalized[file_id] = value

    if not normalized:
        raise ValueError("files_map.json is empty")

    return normalized


def postprocess(output_dir):
    raw_path = os.path.join(output_dir, "raw_corpus.bin")
    meta_path = os.path.join(output_dir, "sentences_meta.bin")
    files_map_path = os.path.join(output_dir, "files_map.json")

    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"Missing raw_corpus.bin: {raw_path}")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing sentences_meta.bin: {meta_path}")

    files_map = load_and_normalize_files_map(files_map_path)

    raw_tmp_path = raw_path + ".tmp"
    meta_tmp_path = meta_path + ".tmp"
    map_tmp_path = files_map_path + ".tmp"

    total_sentences_written = 0
    char_offset = 0
    per_file_line_number = {}

    with (
        open(raw_path, "rb") as raw_in,
        open(meta_path, "rb") as meta_in,
        open(raw_tmp_path, "wb") as raw_out,
        open(meta_tmp_path, "wb") as meta_out,
    ):
        while True:
            raw_line = raw_in.readline()
            if not raw_line:
                break

            meta_chunk = meta_in.read(META_STRUCT.size)
            if len(meta_chunk) != META_STRUCT.size:
                raise ValueError(
                    "sentences_meta.bin has fewer records than raw_corpus.bin lines"
                )

            file_id, _, _, _ = META_STRUCT.unpack(meta_chunk)
            if file_id not in files_map:
                raise ValueError(
                    f"file_id {file_id} from metadata is missing in files_map.json"
                )

            text_line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not text_line.strip():
                continue

            for sentence in split_to_sentences(text_line):
                sentence_bytes = trim_utf8_to_uint16(sentence)
                if not sentence_bytes:
                    continue

                line_number = per_file_line_number.get(file_id, 0) + 1
                if line_number > MAX_UINT32:
                    raise ValueError(
                        f"line_number exceeds uint32 range for file_id {file_id}"
                    )

                text_length = len(sentence_bytes)
                raw_out.write(sentence_bytes)
                raw_out.write(b"\n")
                meta_out.write(
                    META_STRUCT.pack(
                        file_id,
                        line_number,
                        char_offset,
                        text_length,
                    )
                )

                per_file_line_number[file_id] = line_number
                char_offset += text_length + 1
                total_sentences_written += 1

        if meta_in.read(1):
            raise ValueError(
                "sentences_meta.bin has more records than raw_corpus.bin lines"
            )

    sorted_files_map = {
        str(file_id): files_map[file_id]
        for file_id in sorted(files_map.keys())
    }
    with open(map_tmp_path, "w", encoding="utf-8") as handle:
        json.dump(sorted_files_map, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    os.replace(raw_tmp_path, raw_path)
    os.replace(meta_tmp_path, meta_path)
    os.replace(map_tmp_path, files_map_path)

    raw_size = os.path.getsize(raw_path)
    meta_size = os.path.getsize(meta_path)
    is_meta_aligned = (meta_size % META_STRUCT.size) == 0

    print(f"Total sentences written: {total_sentences_written}")
    print(f"raw_corpus.bin size (bytes): {raw_size}")
    print(f"sentences_meta.bin size (bytes): {meta_size}")
    print(f"sentences_meta.bin size multiple of 16 bytes: {is_meta_aligned}")
    print(f"files_map.json entries: {len(sorted_files_map)}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "processed_data")
    postprocess(output_dir)


if __name__ == "__main__":
    main()
