import re

from Tokenizer import SimpleTokenizerV1, SimpleTokenizerV2


def main():
    with open("./the-verdict.txt", "r", encoding="utf-8") as file:
        content = file.read()

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', content)
    preprocessed = [word.strip() for word in preprocessed if word.strip()]

    all_words = sorted(set(preprocessed))
    all_words.extend(["<|endoftext|>", "<|unk|>"])
    vocab_size = len(all_words)

    vocab = {word: index + 1000 for index, word in enumerate(all_words)}

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."

    text = " <|endoftext|> ".join((text1, text2))

    tokenizer = SimpleTokenizerV2(vocab)

    ids = tokenizer.encode(text)
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
