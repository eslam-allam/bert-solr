import numpy as np
import pickle
from io import IOBase, BufferedReader, BytesIO
import requests
import sys

query = sys.argv[1]
REQUEST_FILE = "./request.fifo"
REPLY_FILE = "./reply.fifo"


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


with open(REQUEST_FILE, "wb") as f:
    send_embedding_request([query], f)

with open(REPLY_FILE, "rb") as f:
    embedded_query = recieve_embeddings(f)
embedded_query = [float(x) for x in embedded_query[0]]

solr_query = f"{{!knn f=title_bert_vector topK=10}}{embedded_query}"
params = {"q": solr_query, "wt": "json", "fl": "original_dc_title score"}
response = requests.post(
    "http://localhost:8984/solr/adri_documents/query",
    json={"params": params},
    headers={"Content-type": "application/json"},
)

if response.status_code != 200:
    print(response.text)
else:
    sep = "-" * 30
    result = f"Query: {query}\n\nSearch results:\n\n"
    for i, doc in enumerate(response.json()["response"]["docs"]):
        result += f'\n{sep}\n{i+1}. {doc["original_dc_title"]}\nScore: {doc["score"]}\n{sep}\n'
    print(result)
