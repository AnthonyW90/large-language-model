from bpe_tokenizer import train_bpe_tokenizer, test_tokenizer


def main():
    """Main entry point for LLM exploration."""
    # Example: Train and test BPE tokenizer
    tokenizer = train_bpe_tokenizer(
        gutenberg_dir="../gutenberg/data/text",
        num_files=10,
        vocab_size=250,
        tokenizer_path="./bpe_tokenizer/bpe_tokenizer.pkl",
    )

    test_tokenizer(tokenizer)


if __name__ == "__main__":
    main()
