from io import BytesIO
from random import choice
from string import ascii_letters
import numpy as np

NUMBER_OF_WORDS = 100_000
CHARS_PER_WORD = 12

with open("./test-bert.fifo", "w", encoding="utf-8") as f:
    words = (
        "".join((choice(ascii_letters) for _ in range(CHARS_PER_WORD)))
        for _ in range(NUMBER_OF_WORDS)
    )
    new_words = []

    for i, word in enumerate(words):
        if i % CHARS_PER_WORD == 0:
            new_words.append(f"{word}.")
        else:
            new_words.append(word)
    string = " ".join(new_words)
    print(f"Payload content: '{string}'")
    print(f"Byte size of payload: {len(string.encode('utf-8'))}")
    print(f"Number of words in payload: {len(string.split(' '))}")
    f.write(string)
with open("./out.fifo", "rb") as fs:
    print("Recieved payload embeddings...")
    buffer = BytesIO(fs.read())
    buffer.seek(0)
    print(f"Generated embeddings length: {len( np.load(buffer) )}")
