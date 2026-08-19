import re
import pickle
import time
from collections import defaultdict

def clean_word(word):
    word = word.lower()
    return re.sub(r'[^a-z0-9]', '', word)

def generate_ngrams(word, n):
    ngrams = []
    if len(word) >= n:
        for i in range(len(word) - n + 1):
            ngrams.append(word[i:i+n])
    return ngrams

def build_offline_data():
    print("Starting Offline Processing for Real Data (this might take a few minutes...)")
    start_time = time.time()
    
    dict_2 = defaultdict(set)
    dict_3 = defaultdict(set)
    
    file_path = "real_data/raw_corpus.bin" # The text file sent by the friend
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_id, line in enumerate(f):
            if line_id % 50000 == 0:
                print(f"Processed {line_id} lines...")
                
            words = line.split()
            for word in words:
                cleaned = clean_word(word)
                if not cleaned:
                    continue
                    
                # 2-grams
                for ngram in generate_ngrams(cleaned, 2):
                    dict_2[ngram].add(line_id)
                    
                # 3-grams
                for ngram in generate_ngrams(cleaned, 3):
                    dict_3[ngram].add(line_id)
                        
    print(f"Finished parsing text in {time.time() - start_time:.2f} seconds.")
    
    print("Saving 2_grams.bin...")
    with open("real_data/2_grams.bin", "wb") as f:
        pickle.dump(dict_2, f)
        
    print("Saving 3_grams.bin...")
    with open("real_data/3_grams.bin", "wb") as f:
        pickle.dump(dict_3, f)
        
    print("Offline processing complete! Binary files are ready in 'real_data/' folder.")

if __name__ == "__main__":
    build_offline_data()
