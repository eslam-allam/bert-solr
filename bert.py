import time as t
from io import BytesIO

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

path = "./test-pipe"

print("Loading model...")
start = t.perf_counter()
device = torch.device("cuda")
model = SentenceTransformer("bert-base-nli-mean-tokens").to(device)
print(f"Model loaded in: {t.perf_counter() - start}")

while True:
    with open("./test-bert.fifo", "r", encoding="utf-8") as f:
        data = f.readlines()
        data = (x.removesuffix("\n") for x in "".join(data).split("."))
        data = [x for x in data if x]
        print(f"Number of sentences: {len(data)}")
        print("Generating embeddings...")
        start = t.perf_counter()
        embedding: np.ndarray = model.encode(
            data, device=device, show_progress_bar=True
        )
        print(f"Total embedding length: {len(embedding)}")
        print(f"Embeddings generated in: {t.perf_counter() - start}")
    with open("./out.fifo", "wb") as fd:
        buffer = BytesIO()
        np.save(buffer, embedding)
        buffer.seek(0)
        fd.write(buffer.read())
