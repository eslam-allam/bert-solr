import time as t
from io import BytesIO

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import pickle


REQUEST_FILE = "request.fifo"
REPLY_FILE = "reply.fifo"


print("Loading model...")
start = t.perf_counter()
device = torch.device("cuda")
model = SentenceTransformer("bert-base-nli-mean-tokens").to(device)
print(f"Model loaded in: {t.perf_counter() - start}")

while True:
    with open(REQUEST_FILE, "rb") as f:
        buffer = BytesIO(f.read())
        buffer.seek(0)
        data: list[str] = pickle.load(buffer)
        print(f"Number of sentences: {len(data)}")
        print("Generating embeddings...")
        start = t.perf_counter()
        embedding = model.encode(data, device=device, show_progress_bar=True)
        print(f"Total embedding length: {len(embedding)}")
        print(f"Embeddings generated in: {t.perf_counter() - start}")
    with open(REPLY_FILE, "wb") as fd:
        buffer = BytesIO()
        np.save(buffer, embedding)
        buffer.seek(0)
        fd.write(buffer.read())
