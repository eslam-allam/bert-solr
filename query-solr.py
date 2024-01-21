from enum import Enum
import numpy as np
import pickle
from io import IOBase, BufferedReader, BytesIO
import requests
from tabulate import tabulate
from itertools import zip_longest
import termios
import sys
import tty
import os
from utils.console_utils import Loader

class QueryType(Enum):
    EDISMAX = 'edismax'
    VECTOR = 'vector'
    HYBRID = 'hybrid'

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

def generate_vector_query(query: str, page: int, rows: int, request_file: str, reply_file: str):
    with open(request_file, "wb") as f:
        send_embedding_request([query], f)

    with open(reply_file, "rb") as f:
        embedded_query = recieve_embeddings(f)
    embedded_query = [float(x) for x in embedded_query[0]]

    return f"{{!knn f=title_bert_vector topK={(page + 1) * rows}}}{embedded_query}"


def query_solr(query_type: QueryType, query: str, request_file: str, reply_file: str, page: int, rows: int):
    edismax_query = f"{{!edismax qf='original_dc_title^5 original_dc_description_abstract^2'}}{query}"
    rerank_query = "{!rerank reRankQuery=$rqq reRankDocs=50 reRankWeight=3}"


    params = {"wt": "json", "fl": "original_dc_title score", "start": page * rows, "rows": rows}

    if query_type == QueryType.EDISMAX:
        params["q"] = edismax_query
    elif query_type == QueryType.VECTOR:
        vector_query = generate_vector_query(query, page, rows, request_file, reply_file)
        params["q"] = vector_query
    elif query_type == QueryType.HYBRID:
        vector_query = generate_vector_query(query, page, rows, request_file, reply_file)
        params["q"] = vector_query
        params["rqq"] = edismax_query
        params["rq"] = rerank_query

    response = requests.post(
        "http://localhost:8984/solr/adri_documents/query",
        json={"params": params},
        headers={"Content-type": "application/json"},
    )

    results = {"title": [], "score": []}
    if response.status_code != 200:
        print(response.text)
        return results
    else:
        for doc in response.json()["response"]["docs"]:
            results["title"].append(doc['original_dc_title'])
            results['score'].append(doc['score'])
        return results

def build_result(request_file: str, reply_file: str, query: str, page:int, rows: int):
    return lambda query_type: query_solr(query_type, query, request_file, reply_file, page, rows)

def slice_value(value, start, end):
    if start > len(value) - 1:
        return value[-1]
    if end >= len(value):
        return value[start:]
    return value[start:end]



def concat_lists(label1: str, label2: str, label3: str, items1, items2, items3, max_width: int | None ):
    values: list[str] = []
    for e, v, h in zip_longest(items1, items2, items3):
        value = []
        if e is not None:
            label = f'{label1}: {e}'
            if max_width is None:
                max_width = len(label)
            value.append('\n'.join([ slice_value(label, i, i+max_width) for i in range(0, len(label), max_width) ]))

        if v is not None:
            label = f'{label2}: {v}'
            if max_width is None:
                max_width = len(label)
            value.append('\n'.join( [ slice_value(label, i, i+max_width) for i in range(0, len(label), max_width) ] ))
        if h is not None:
            label = f'{label3}: {h}'
            if max_width is None:
                max_width = len(label)
            value.append('\n'.join( [ slice_value(label, i, i+max_width) for i in range(0, len(label), max_width) ] ))

        
        values.append('\n\n'.join(value))

    return values

def concat_lists_builder(label1: str, label2: str, label3: str, max_width: int | None = None):
    return lambda items1, items2, items3: concat_lists(label1, label2, label3, items1, items2, items3, max_width)

def getkey():
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        while True:
            b = os.read(sys.stdin.fileno(), 3).decode()
            if len(b) == 3:
                k = ord(b[2])
            else:
                k = ord(b)
            key_mapping = {
                127: 'backspace',
                10: 'return',
                32: 'space',
                9: 'tab',
                27: 'esc',
                65: 'up',
                66: 'down',
                67: 'right',
                68: 'left'
            }
            return key_mapping.get(k, chr(k))
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
if __name__ == "__main__":

    REQUEST_FILE = "./request.fifo"
    REPLY_FILE = "./reply.fifo"


    page = 0
    last_index = 0
    print("\033[H\033[J", end="")
    query = input("Enter search query: ")
    rows = input("How many documents should be returned per page? [default: 10]: ")
    if not rows:
        rows = 10
    rows = int(rows)
    print("\033[H\033[J", end="")
    try:
        while True:
            print("\033[H\033[J", end="")
            data = {}
            result_builder = build_result(REQUEST_FILE, REPLY_FILE, query, page, rows)
            edismax_result = {'title': [], 'score': []} 
            vector_result = {'title': [], 'score': []}
            hybrid_result = {'title': [], 'score': []}

            with Loader("Fetching documents..."):
                edismax_result = result_builder(QueryType.EDISMAX)
                vector_result = result_builder(QueryType.VECTOR)
                hybrid_result = result_builder(QueryType.HYBRID)

        
            concater = concat_lists_builder("edismax", "vector", "hybrid", 120)

            max_results = max((len(edismax_result['title']), len(vector_result['title']), len(hybrid_result['title'])))
            max_index =  max_results + last_index
            data['#'] = [ x + 1 for x in range(last_index, max_index)]
            data["title"] = concater(edismax_result['title'], vector_result['title'], hybrid_result['title'])
            data["score"] = concater(edismax_result['score'], vector_result['score'], hybrid_result['score'])


            print(f"\nQuery: {query}")
            print(tabulate(data, headers="keys", tablefmt='fancy_grid'))

            print("Enter [Enter] for next page, [Backspace] for previous page, [Tab] to change query, [Ctrl+c] to abort...")

            while True:
                key = getkey()
                if key == 'backspace':
                    if page == 0:
                        print("Already at first page!!!")
                    else:
                        page -= 1
                        last_index = last_index - max_results
                        break
                elif key == 'return':
                    page += 1
                    last_index = max_index
                    break
                elif key == 'tab':
                    query = input("Enter new Query: ")
                    page = 0
                    last_index = 0
                    break
                else:
                    print("Invalid key...")
    except KeyboardInterrupt:
        print("Exitting...")


