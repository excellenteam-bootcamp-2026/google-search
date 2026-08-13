import pickle
from typing import Dict, Set, List

class BinaryDataLoader:
    def __init__(self, master_text_path: str, bigrams_path: str, trigrams_path: str):
        self.master_text_path = master_text_path
        self.bigrams_path = bigrams_path
        self.trigrams_path = trigrams_path
        
        # In-memory databases
        self.original_text_db: List[str] = []
        self.ngram_index: Dict[str, Set[int]] = {}
        
    def load_data(self):
        print("Loading offline data into memory...")
        
        # Load master text file
        # Each line index corresponds to the ID
        with open(self.master_text_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.original_text_db.append(line.rstrip('\n'))
                
        # Load binary n-gram files
        with open(self.bigrams_path, 'rb') as f:
            bigrams = pickle.load(f)
            self.ngram_index.update(bigrams)
            
        with open(self.trigrams_path, 'rb') as f:
            trigrams = pickle.load(f)
            self.ngram_index.update(trigrams)
            
        print("Data loaded successfully.")

    def get_sentence(self, sentence_id: int) -> str:
        if 0 <= sentence_id < len(self.original_text_db):
            return self.original_text_db[sentence_id]
        return ""

    def get_sentence_ids(self, ngram: str) -> Set[int]:
        return self.ngram_index.get(ngram, set())
