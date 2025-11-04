import re

from Tokenizer import SimpleTokenizerV1


def main():
    with open("./the-verdict.txt", "r", encoding="utf-8") as file:
        content = file.read()

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', content)
    preprocessed = [word.strip() for word in preprocessed if word.strip()]

    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)

    vocab = {word: index + 1000 for index, word in enumerate(all_words)}

    tokenizer = SimpleTokenizerV1(vocab)

    text = content[:99]

    ids = tokenizer.encode(text)

    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
