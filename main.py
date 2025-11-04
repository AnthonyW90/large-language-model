import tiktoken


def main():
    with open("./the-verdict.txt", "r", encoding="utf-8") as file:
        content = file.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(content)
    print(tokens[:99])


if __name__ == "__main__":
    main()
