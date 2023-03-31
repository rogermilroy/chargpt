import os

project_base_dir = os.path.dirname(os.path.abspath(__file__))

data_filename = "data/input.txt"


def get_vocab(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f'Vocab: {"".join(chars)}')
    print(f"Vocab size: {vocab_size}")


def entrypoint():
    with open(os.path.join(project_base_dir, data_filename), "r", encoding="utf8") as f:
        text = f.read()
        print(f"Total text len: {len(text)}")
        get_vocab(text)


if __name__ == "__main__":
    entrypoint()
