import re
from collections import Counter
from typing import List

from models import AutoCompleteData
from data_loader import BinaryDataLoader
from scoring import Scoring

class AutoCompleteEngine:
    def __init__(self, data_loader: BinaryDataLoader):
        self.data_loader = data_loader
        
    def _clean_query(self, query: str) -> str:
        # Lowercase and remove anything that is not a-z, 0-9, or space
        query = query.lower()
        return re.sub(r'[^a-z0-9\s]', '', query)
        
    def _generate_ngrams(self, cleaned_query: str) -> List[str]:
        ngrams = []
        words = cleaned_query.split()
        for word in words:
            # Generate 2-grams
            if len(word) >= 2:
                for i in range(len(word) - 1):
                    ngrams.append(word[i:i+2])
            # Generate 3-grams
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    ngrams.append(word[i:i+3])
        return ngrams
        
    def _get_top_k_candidates(self, ngrams: List[str], k: int = 100) -> List[int]:
        if not ngrams:
            return []
            
        candidate_counter = Counter()
        for ngram in ngrams:
            ids = self.data_loader.get_sentence_ids(ngram)
            # Add occurrence for each ID found
            for sentence_id in ids:
                candidate_counter[sentence_id] += 1
                
        # Return the top k IDs that appeared in the most ngrams
        top_k = candidate_counter.most_common(k)
        return [sentence_id for sentence_id, count in top_k]

    def get_best_k_completions(self, prefix: str, k: int = 5) -> List[AutoCompleteData]:
        clean_prefix = self._clean_query(prefix)
        if not clean_prefix:
            return []
            
        ngrams = self._generate_ngrams(clean_prefix)
        
        # If the query is so short it has no 2-grams or 3-grams, we can't search effectively in this model.
        # As an edge case, we could just return empty or search the first lines, but usually
        # users type at least 2 chars.
        if not ngrams:
            return []
            
        # 1. Filter: Get top candidate sentence IDs
        candidate_ids = self._get_top_k_candidates(ngrams, k=150)
        
        completions = []
        
        # 2. Verify: Score each candidate
        for sentence_id in candidate_ids:
            sentence = self.data_loader.get_sentence(sentence_id)
            if not sentence:
                continue
                
            score = Scoring.calculate_score(prefix, sentence)
            
            # If score > 0, it means it's a valid completion (max 1 error)
            if score > 0:
                completions.append(
                    AutoCompleteData(
                        completed_sentence=sentence,
                        source_text=self.data_loader.master_text_path, # Dynamically get the source file path
                        offset=sentence_id,            # The ID is the line number
                        score=score
                    )
                )
                
        # 3. Rank: Sort by Score (Desc) -> Alphabetical (Asc)
        completions.sort(key=lambda x: (-x.score, x.completed_sentence))
        
        return completions[:k]
