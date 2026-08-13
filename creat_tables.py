import os
import re
import struct
from collections import defaultdict

# נתיבים
BASE_DIR = "/mnt/c/Users/shire/Desktop/google-search/processed_data"
CORPUS_PATH = os.path.join(BASE_DIR, "raw_corpus.bin")


def normalize_text(text: str) -> str:
    """נורמליזציה: אותיות קטנות, השארת a-z, 0-9 ורווחים בלבד."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_indices():
    bigram_index = defaultdict(list)
    trigram_index = defaultdict(list)

    print("Reading corpus and extracting n-grams...")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for sentence_id, line in enumerate(f):
            cleaned_text = normalize_text(line)

            # חילוץ ביגרמים ייחודיים למשפט הנוכחי
            unique_bigrams = set()
            for i in range(len(cleaned_text) - 1):
                unique_bigrams.add(cleaned_text[i : i + 2])

            for bg in unique_bigrams:
                bigram_index[bg].append(sentence_id)

            # חילוץ טריגרמים ייחודיים למשפט הנוכחי
            unique_trigrams = set()
            for i in range(len(cleaned_text) - 2):
                unique_trigrams.add(cleaned_text[i : i + 3])

            for tg in unique_trigrams:
                trigram_index[tg].append(sentence_id)

    # כתיבת Bigrams
    write_index(
        index_data=bigram_index,
        dict_path=os.path.join(BASE_DIR, "bigram_dict.bin"),
        postings_path=os.path.join(BASE_DIR, "bigram_postings.bin"),
        struct_fmt="<2s2xIQ",  # 2 chars + 2 padding + uint32 + uint64 = 16 bytes
        n_size=2,
    )

    # כתיבת Trigrams
    write_index(
        index_data=trigram_index,
        dict_path=os.path.join(BASE_DIR, "trigram_dict.bin"),
        postings_path=os.path.join(BASE_DIR, "trigram_postings.bin"),
        struct_fmt="<3s1xIQ",  # 3 chars + 1 padding + uint32 + uint64 = 16 bytes
        n_size=3,
    )


def write_index(index_data, dict_path, postings_path, struct_fmt, n_size):
    print(f"Writing index files: {dict_path} and {postings_path}...")

    # מיוון אלפביתי של המילון
    sorted_keys = sorted(index_data.keys())

    current_posting_offset = 0

    with (
        open(dict_path, "wb") as f_dict,
        open(postings_path, "wb") as f_postings,
    ):
        for key in sorted_keys:
            postings_list = index_data[key]
            posting_count = len(postings_list)

            # 1. כתיבת רשימת המשפטים ב-Postings (כל ID נשמר כ-uint32)
            postings_bytes = struct.pack(f"<{posting_count}I", *postings_list)
            f_postings.write(postings_bytes)

            # 2. כתיבת הרשומה ב-Dictionary (גודל קבוע: 16 בייטים)
            key_bytes = key.encode("ascii")
            dict_entry = struct.pack(
                struct_fmt,
                key_bytes,
                posting_count,
                current_posting_offset,
            )
            f_dict.write(dict_entry)

            # עדכון ה-Offset לקראת המפתח הבא (נמדד בבייטים)
            current_posting_offset += posting_count * 4

    print(
        f"Done. Indexed {len(sorted_keys)} unique {n_size}-grams successfully."
    )


if __name__ == "__main__":
    build_indices()