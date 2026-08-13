import string

def get_base_score(query: str) -> int:
    """
    Base score: 2 x the number of case-insensitive matching letters including space but not punctuation.
    """
    valid_chars = 0
    for char in query:
        if char not in string.punctuation:
            valid_chars += 1
    return valid_chars * 2

class Scoring:
    @staticmethod
    def calculate_score(query: str, sentence: str) -> int:
        """
        Finds the best matching substring in 'sentence' for the given 'query'.
        Tolerates at most 1 error (replacement, deletion, addition).
        Returns the highest possible score, or 0 if more than 1 error is required.
        """
        # Clean both query and sentence for comparison, keeping only alphanumeric and spaces
        clean_q = ''.join(c for c in query.lower() if c.isalnum() or c == ' ')
        clean_s = ''.join(c for c in sentence.lower() if c.isalnum() or c == ' ')
        
        base_score = get_base_score(query)
        best_score = 0
        
        # Edge case: Query is much longer than the sentence
        if len(clean_q) > len(clean_s) + 1:
            return 0
            
        # Try to find the query in the sentence with a sliding window
        # Window sizes to check: len(q), len(q)-1, len(q)+1
        
        # 1. Exact match / Replacement check (window = len(q))
        q_len = len(clean_q)
        for i in range(len(clean_s) - q_len + 1):
            window = clean_s[i:i+q_len]
            if window == clean_q:
                return base_score # Perfect match, max score, return immediately
            
            # Check replacement
            diffs = [(idx, q_char, w_char) for idx, (q_char, w_char) in enumerate(zip(clean_q, window)) if q_char != w_char]
            if len(diffs) == 1:
                err_idx = diffs[0][0]
                deduction = 5 - err_idx if err_idx < 4 else 1
                best_score = max(best_score, base_score - deduction)
                
        # 2. Deletion in query (query has an extra char, so window = len(q) - 1)
        if q_len > 1:
            for i in range(len(clean_s) - (q_len - 1) + 1):
                window = clean_s[i:i+(q_len - 1)]
                # Check if removing one char from clean_q makes it match window
                for err_idx in range(q_len):
                    temp_q = clean_q[:err_idx] + clean_q[err_idx+1:]
                    if temp_q == window:
                        deduction = 10 - (2 * err_idx) if err_idx < 4 else 2
                        best_score = max(best_score, base_score - deduction)
                        
        # 3. Addition in query (query is missing a char, so window = len(q) + 1)
        if q_len + 1 <= len(clean_s):
            for i in range(len(clean_s) - (q_len + 1) + 1):
                window = clean_s[i:i+(q_len + 1)]
                # Check if adding one char to clean_q makes it match window
                # Which is equivalent to checking if removing one char from window makes it match clean_q
                for err_idx in range(len(window)):
                    temp_w = window[:err_idx] + window[err_idx+1:]
                    if temp_w == clean_q:
                        # The error index is relative to the query string that the user typed.
                        # Since the query is missing the character at err_idx, the penalty is based on err_idx.
                        deduction = 10 - (2 * err_idx) if err_idx < 4 else 2
                        best_score = max(best_score, base_score - deduction)
                        
        return best_score
