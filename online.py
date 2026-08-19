from data_loader import BinaryDataLoader
from search_engine import AutoCompleteEngine

def main():
    print("Initializing Google Search Autocomplete System...")
    
    # Initialize the data loader with the paths to our binary files and master text
    loader = BinaryDataLoader(
        master_text_path="real_data/raw_corpus.bin", # The huge real text file
        bigrams_path="real_data/2_grams.bin",
        trigrams_path="real_data/3_grams.bin"
    )
    
    loader.load_data()
    engine = AutoCompleteEngine(loader)
    
    print("\nThe system is ready. Enter your text:")
    
    current_input = ""
    
    while True:
        try:
            # We append input so the user can continue typing, as described in the PDF
            user_input = input(current_input)
            
            if user_input.strip() == "#":
                current_input = ""
                print("\n--- Resetting input ---")
                print("Enter your text:")
                continue
                
            current_input += user_input
            
            # Get completions
            completions = engine.get_best_k_completions(current_input, k=5)
            
            if completions:
                print(f"\nHere are {len(completions)} suggestions")
                for i, c in enumerate(completions, 1):
                    print(f"{i}. {c.completed_sentence} (Score: {c.score})")
            else:
                print("\nNo suggestions found.")
                
            print("\n", end="")
            
        except KeyboardInterrupt:
            print("\nExiting system. Goodbye!")
            break

if __name__ == "__main__":
    main()
