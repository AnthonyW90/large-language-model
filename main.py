import os
import pickle
from pathlib import Path
from Tokenizer import BPETokenizer


def load_gutenberg_files(data_dir, limit=None):
    """
    Load text files from the gutenberg directory.

    Args:
        data_dir: Path to the directory containing text files
        limit: Maximum number of files to load (None for all)

    Returns:
        List of text contents from the files
    """
    data_path = Path(data_dir)

    # Get all text files matching the pattern PG*_text.txt
    text_files = sorted(data_path.glob("PG*_text.txt"))

    if limit:
        text_files = text_files[:limit]

    print(f"Loading {len(text_files)} files from {data_dir}...")

    corpus = []
    for i, file_path in enumerate(text_files, 1):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                corpus.append(content)

            if i % 100 == 0:
                print(f"  Loaded {i}/{len(text_files)} files...")
        except Exception as e:
            print(f"  Error loading {file_path.name}: {e}")

    print(f"Successfully loaded {len(corpus)} files")
    return corpus


def save_tokenizer(tokenizer, filepath="bpe_tokenizer.pkl"):
    """Save the trained tokenizer to disk."""
    with open(filepath, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {filepath}")


def load_tokenizer(filepath="bpe_tokenizer.pkl"):
    """Load a trained tokenizer from disk."""
    with open(filepath, "rb") as f:
        tokenizer = pickle.load(f)
    print(f"Tokenizer loaded from {filepath}")
    return tokenizer


def main():
    # Configuration
    gutenberg_dir = "../gutenberg/data/text"
    num_files = 100  # None = all files, or set a number to limit
    vocab_size = 10000  # Increased for larger corpus
    tokenizer_path = "bpe_tokenizer.pkl"

    # Check if we already have a trained tokenizer
    if os.path.exists(tokenizer_path):
        print(f"Found existing tokenizer at {tokenizer_path}")
        response = input(
            "Do you want to (l)oad it or (t)rain a new one? [l/t]: "
        ).lower()

        if response == "l":
            tokenizer = load_tokenizer(tokenizer_path)
        else:
            print("\nTraining new tokenizer...")
            corpus = load_gutenberg_files(gutenberg_dir, limit=num_files)

            if not corpus:
                print("No files loaded. Please check the directory path.")
                return

            print(f"\nTraining BPE tokenizer with vocab_size={vocab_size}...")
            tokenizer = BPETokenizer(vocab_size=vocab_size)
            tokenizer.train(corpus)

            # Save the trained tokenizer
            save_tokenizer(tokenizer, tokenizer_path)
    else:
        # No existing tokenizer, train a new one
        corpus = load_gutenberg_files(gutenberg_dir, limit=num_files)

        if not corpus:
            print("No files loaded. Please check the directory path.")
            return

        print(f"\nTraining BPE tokenizer with vocab_size={vocab_size}...")
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        tokenizer.train(corpus)

        # Save the trained tokenizer
        save_tokenizer(tokenizer, tokenizer_path)

    # Test the tokenizer
    print("\n" + "=" * 50)
    print("Testing the tokenizer:")
    print("=" * 50)

    test_strings = [
        "Learning LLMs is fun!",
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing is fascinating.",
    ]

    for test_string in test_strings:
        tokens = tokenizer.tokenize(test_string)
        token_ids = tokenizer.encode(test_string)
        decoded = tokenizer.decode(token_ids)

        print(f"\nOriginal: '{test_string}'")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}")
        print(f"Decoded: '{decoded}'")

    # Show vocabulary statistics
    print("\n" + "=" * 50)
    print("Vocabulary Statistics:")
    print("=" * 50)
    print(f"Total vocabulary size: {len(tokenizer.vocab)}")
    print(f"Total merges learned: {len(tokenizer.merges)}")


if __name__ == "__main__":
    main()
