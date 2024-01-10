import numpy as np
import pickle
from io import IOBase, BufferedReader, BytesIO
import requests
import json


def send_embedding_request(sentences: list[str], medium: IOBase):
    if medium.readable():
        print("Medium must be write only..")
        return False
    if any([not x for x in sentences]):
        print("Cannot embed empty text.")
        return False

    pickle.dump(sentences, medium)
    return True


def recieve_embeddings(recieve_file: BufferedReader) -> np.ndarray:
    if recieve_file.writable():
        print("Recieve file must be read only.")
        return np.ndarray((1,))
    buffer = BytesIO(recieve_file.read())
    buffer.seek(0)
    return np.load(buffer)


QUERY = "Artificial Inteligience"
with open("./test-bert.fifo", "wb") as f:
    send_embedding_request([QUERY], f)

with open("./out.fifo", "rb") as f:
    embedded_query = recieve_embeddings(f)
embedded_query = [float(x) for x in embedded_query[0]]

solr_query = f"{{!knn f=title_bert_vector topK=10}}{embedded_query}"
params = {"q": solr_query, "wt": "json"}
response = requests.post(
    "http://localhost:8983/solr/adri_documents/query",
    json={"params": params},
    headers={"Content-type": "application/json"},
)

if response.status_code != 200:
    print(response.text)
else:
    print(
        json.dumps(
            {
                f"doc_{i + 1}": x["original_dc_title"]
                for i, x in enumerate(response.json()["response"]["docs"])
            }
        )
    )
