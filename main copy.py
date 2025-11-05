from Tokenizer import BPETokenizer


def main():
    with open("./the-verdict.txt", "r", encoding="utf-8") as file:
        content = file.read()

    tokenizer = BPETokenizer()
    tokenizer.train(content.split("\n"))

    test_string = "Learning LLMs is fun!"

    print(tokenizer.tokenize(test_string))
    print(tokenizer.encode(test_string))
    print(tokenizer.decode(tokenizer.encode(test_string)))


if __name__ == "__main__":
    main()
