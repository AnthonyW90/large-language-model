import re


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [word.strip() for word in preprocessed if word.strip()]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.:;?_!"()\'])', r"\1", text)
        return text


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.:;?_!"()\'])', r"\1", text)
        return text


"""
Basic Byte Pair Encoding (BPE) Tokenizer Implementation

This implementation demonstrates the core concepts of BPE tokenization:
1. Start with a vocabulary of individual characters
2. Iteratively merge the most frequent pair of tokens
3. Build a vocabulary of subword units
"""

from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import re


class BPETokenizer:
    def __init__(self, vocab_size: int = 1000):
        """
        Initialize the BPE tokenizer.

        Args:
            vocab_size: Target vocabulary size (number of merge operations + base chars)
        """
        self.vocab_size = vocab_size
        self.merges = {}  # Maps pair -> merged token
        self.vocab = {}  # Maps token -> index

    def get_stats(self, words: Dict[Tuple[str, ...], int]) -> Counter:
        """
        Count frequency of adjacent pairs in the corpus.

        Args:
            words: Dictionary mapping word tuples to their frequencies

        Returns:
            Counter of pair frequencies
        """
        pairs = Counter()
        for word, freq in words.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def merge_pair(
        self, pair: Tuple[str, str], words: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, ...], int]:
        """
        Merge all occurrences of a pair in the vocabulary.

        Args:
            pair: The pair to merge
            words: Current word representations

        Returns:
            Updated words dictionary with merged pairs
        """
        new_words = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)

        for word in words:
            new_word = []
            i = 0
            while i < len(word):
                # Check if current position matches the pair to merge
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words[tuple(new_word)] = words[word]

        return new_words

    def train(self, corpus: List[str]):
        """
        Train the BPE tokenizer on a corpus.

        Args:
            corpus: List of text strings to train on
        """
        # Preprocess: split into words and count frequencies
        word_freqs = Counter()
        for text in corpus:
            # Simple whitespace tokenization
            words = text.split()
            word_freqs.update(words)

        # Convert words to character sequences with end-of-word marker
        words = {}
        for word, freq in word_freqs.items():
            # Split into characters and add end marker
            words[tuple(list(word) + ["</w>"])] = freq

        # Get initial vocabulary (all characters)
        base_vocab = set()
        for word in words:
            base_vocab.update(word)

        # Determine number of merges needed
        num_merges = self.vocab_size - len(base_vocab)

        # Perform BPE merges
        for i in range(num_merges):
            pairs = self.get_stats(words)
            if not pairs:
                break

            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)

            # Merge the pair
            words = self.merge_pair(best_pair, words)

            # Store the merge operation
            self.merges[best_pair] = "".join(best_pair)

            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1} merges...")

        # Build final vocabulary
        vocab_tokens = set()
        for word in words:
            vocab_tokens.update(word)

        self.vocab = {token: idx for idx, token in enumerate(sorted(vocab_tokens))}
        print(f"Training complete! Vocabulary size: {len(self.vocab)}")

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using learned BPE merges.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens
        """
        words = text.split()
        tokens = []

        for word in words:
            # Start with character-level representation
            word_tokens = list(word) + ["</w>"]

            # Apply merges
            while len(word_tokens) > 1:
                # Find pairs that can be merged
                pairs = [
                    (word_tokens[i], word_tokens[i + 1])
                    for i in range(len(word_tokens) - 1)
                ]

                # Find which pairs are in our merge rules
                mergeable = [
                    (i, pair) for i, pair in enumerate(pairs) if pair in self.merges
                ]

                if not mergeable:
                    break

                # Merge the first occurrence of the highest priority pair
                # Priority is based on order of learning (earlier = higher priority)
                merge_order = {pair: i for i, pair in enumerate(self.merges.keys())}
                i, pair = min(mergeable, key=lambda x: merge_order[x[1]])

                # Perform the merge
                word_tokens = (
                    word_tokens[:i] + [self.merges[pair]] + word_tokens[i + 2 :]
                )

            tokens.extend(word_tokens)

        return tokens

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab.get("<unk>", 0)) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text
        """
        # Reverse vocabulary lookup
        id_to_token = {idx: token for token, idx in self.vocab.items()}

        tokens = [id_to_token.get(idx, "<unk>") for idx in token_ids]
        text = "".join(tokens).replace("</w>", " ").strip()

        return text


# Example usage
if __name__ == "__main__":
    with open("./the-verdict.txt", "r", encoding="utf-8") as file:
        content = file.read()
    # Sample training corpus
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the dog was lazy and brown",
        "quick foxes are not lazy",
        "the fox jumps high over the dog",
        "brown dogs and brown foxes",
    ]

    # Initialize and train tokenizer
    print("Training BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=150)
    tokenizer.train(corpus)

    print("\n" + "=" * 50)
    print("Learned Merges (first 10):")
    print("=" * 50)
    for i, (pair, merged) in enumerate(list(tokenizer.merges.items())[:10]):
        print(f"{i + 1}. {pair} -> '{merged}'")

    # Test tokenization
    print("\n" + "=" * 50)
    print("Tokenization Examples:")
    print("=" * 50)

    test_texts = ["the quick brown fox", "lazy dogs", "foxes jump"]

    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)

        print(f"\nText: '{text}'")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Decoded: '{decoded}'")
