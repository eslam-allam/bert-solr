import requests
import argparse
import logging
from utils.logging_utils import ColouredLogger
from utils.signal_utils import GracefulKiller
from io import BytesIO, BufferedReader, IOBase, TextIOWrapper
import numpy as np
import json
import pickle

logger: ColouredLogger = logging.getLogger(__name__)


class Solr:
    def __init__(
        self,
        host: str,
        port: int,
        collection: str,
    ) -> None:
        self._host = host
        self._port = port
        self._collection = collection
        self._uri = f"http://{host}:{port}/solr/{collection}"

    class _Cursor:
        def __init__(self, uri, q, buffer_size, cursor, sort, total_scrolled) -> None:
            self._uri = uri
            self._doc_buffer = []
            self._q = q
            self._buffer_size = buffer_size
            self._cursor = cursor
            self._previous_cursor = cursor
            self._sort = sort
            self.total_scrolled = total_scrolled

        def __iter__(self):
            return self

        def __next__(self) -> dict[str, object]:
            self.total_scrolled += 1
            if len(self._doc_buffer) != 0:
                return self._doc_buffer.pop()

            logger.debug(
                f"Buffer ran out of documents. Requesting {self._buffer_size} more documents..."
            )
            params = {
                "q": self._q,
                "rows": self._buffer_size,
                "cursorMark": self._cursor,
                "sort": self._sort,
                "wt": "json",
            }
            response = requests.get(
                f"{self._uri}/select",
                params=params,
            )
            if response.status_code != 200:
                raise StopIteration()

            result = response.json()

            if result["nextCursorMark"] == self._cursor:
                raise StopIteration()

            self._previous_cursor = self._cursor
            self._cursor = result["nextCursorMark"]
            self._doc_buffer.extend(result["response"]["docs"])

            return self._doc_buffer.pop()

        def reset(self):
            self._doc_buffer = []
            self._cursor = "*"
            self._previous_cursor = "*"
            self.total_scrolled = 0

        def save(self, f: TextIOWrapper):
            data = {
                "cursor": self._previous_cursor,
                "buffer_remaining": len(self._doc_buffer),
                "uri": self._uri,
                "buffer_size": self._buffer_size,
                "sort": self._sort,
                "q": self._q,
                "total_scrolled": self.total_scrolled,
            }
            json.dump(data, f, indent=4)

    def atomic_update(self, doc_id: str, updates: dict[str, object]):
        if not updates:
            logger.warning(f"Recieved empty updates for document ID: '{doc_id}'")
            return False
        constructed_updates: dict[str, object] = {"id": doc_id}

        for key, value in updates.items():
            constructed_updates[key] = {"set": value}

        payload = json.dumps([constructed_updates])

        response = requests.post(
            f"{self._uri}/update?commitWithin=1000",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        if response.status_code != 200:
            logger.error("Failed to send atomic update to solr.")
            return False
        return True

    def cursor(
        self, q="*:*", buffer_size=10, sort="id asc", cursorMark="*", total_scrolled=0
    ):
        return Solr._Cursor(self._uri, q, buffer_size, cursorMark, sort, total_scrolled)

    def resumedCursor(self, f: TextIOWrapper):
        data: dict[str, object] = json.load(f)
        uri = str(data.get("uri", ""))
        cursorMark = str(data.get("cursor", ""))
        buffer_remaining = data.get("buffer_remaining", -1)
        buffer_size = data.get("buffer_size", -1)
        sort = str(data.get("sort", ""))
        q = str(data.get("q", ""))
        total_scrolled = data.get("total_scrolled", -1)

        if (
            not isinstance(buffer_remaining, int)
            or not isinstance(buffer_size, int)
            or not isinstance(total_scrolled, int)
        ):
            logger.warning("Malformed cursor checkpoint. Returning new cursor.")
            return self.cursor()

        if (
            not uri
            or not cursorMark
            or buffer_remaining == -1
            or buffer_size == -1
            or total_scrolled == -1
            or not sort
            or not q
        ):
            logger.warning("Malformed cursor checkpoint. Returning new cursor.")
            return self.cursor()

        if uri != self._uri:
            logger.warning("Invalid Solr Uri. Returning new cursor.")
            return self.cursor()

        cursor = self.cursor(q, buffer_size, sort, cursorMark, total_scrolled)
        logger.debug("Created resumed cursor. Fast forwarding to resumed state...")
        fast_forward = buffer_size - buffer_remaining
        logger.info(f"Fast forwarding {fast_forward} documents...")
        for doc in cursor:
            logger.debug(f"Popped ID: '{doc['id']}'.")
            if len(cursor._doc_buffer) == buffer_remaining:
                break
        cursor.total_scrolled = total_scrolled

        return cursor


def send_embedding_request(sentences: list[str], medium: IOBase):
    if medium.readable():
        logger.error("Medium must be write only..")
        return False
    if any([not x for x in sentences]):
        logger.warning("Cannot embed empty text.")
        return False

    pickle.dump(sentences, medium)
    return True


def recieve_embeddings(recieve_file: BufferedReader) -> np.ndarray:
    if recieve_file.writable():
        logger.error("Recieve file must be read only.")
        return np.ndarray((1,))
    buffer = BytesIO(recieve_file.read())
    buffer.seek(0)
    return np.load(buffer)


def main(
    host: str,
    port: int,
    collection: str,
    request_file: str,
    recieve_file: str,
    checkpoint_file: str,
    resume_checkpoint: bool,
    query: str,
    sort: str,
    buffer_size: int,
):
    killer = GracefulKiller()
    solr = Solr(host, port, collection)

    if resume_checkpoint:
        logger.info("Opening checkpoint file...")
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            cursor = solr.resumedCursor(f)
    else:
        logger.debug("No checkpoint selected.")
        cursor = solr.cursor(query, buffer_size, sort)

    updated_count = 0
    failed_count = 0
    for doc in cursor:
        doc_id = doc.get("id", "")
        doc_position = cursor.total_scrolled

        if not isinstance(doc_id, str) or not doc_id:
            logger.warning(
                f"Document at position: {doc_position} does not have a doc ID. Skipping..."
            )
            continue

        logger.info(f"{doc_position}.document ID: {doc_id}")

        title = doc.get("original_dc_title")
        if title is None:
            logger.warning(f"Could not find original title in doc with ID: '{doc_id}'")
            failed_count += 1
            continue

        if not isinstance(title, str):
            logger.error(
                f"Original title field in doc with ID: '{doc_id}' is not a string."
            )
            failed_count += 1
            continue
        with open(request_file, "wb") as fs:
            if not send_embedding_request([title], fs):
                logger.error(
                    f"Could not send embedding request for doc with ID: '{doc_id}'"
                )
                failed_count += 1
                continue
        with open(recieve_file, "rb") as fd:
            embeddings = recieve_embeddings(fd)
            logger.debug(
                f"Generated embeddings length for Doc with ID: '{doc_id}': {len(embeddings)}"
            )

        updates = {}
        updates["title_bert_vector"] = [float(w) for w in embeddings[0]]

        if not solr.atomic_update(doc_id, updates):
            logger.warning(f"Failed to update doc with ID: '{doc_id}'")
            failed_count += 1
            continue
        logger.info(f"Succesfully updated doc with ID: '{doc_id}'")
        updated_count += 1

        if killer.kill_now:
            logger.info("Recieved shutdown signal. Exitting gracefully...")
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                cursor.save(f)
            return (updated_count, failed_count)

    return (updated_count, failed_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Solr Vector Indexer",
        "Converts full text from a Solr collection to dense vector embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )

    parser.add_argument(
        "-r",
        "--request-file",
        required=True,
        type=str,
        help="Path to file used for sending the embeddings request.",
    )
    parser.add_argument(
        "-s",
        "--recieve-file",
        required=True,
        type=str,
        help="Path to file used for recieving the embedding result.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        default=False,
        action="store_true",
        help="Enable debug mode.",
    )

    solr_args = parser.add_argument_group("Solr Args")

    solr_args.add_argument(
        "-t",
        "--host",
        default="localhost",
        type=str,
        help="Host of target Solr instance.",
    )
    solr_args.add_argument(
        "-p", "--port", default=8983, type=int, help="Port of target Solr instance."
    )
    solr_args.add_argument(
        "-c",
        "--collection",
        type=str,
        required=True,
        help="Name of target Solr collection.",
    )
    solr_args.add_argument(
        "-q",
        "--query",
        type=str,
        default="*:*",
        help="Query to select solr documents.",
    )
    solr_args.add_argument(
        "-sr",
        "--sort",
        type=str,
        default="Sort criterial for fetching documents (Must include a unique field).",
        help="Query to select solr documents.",
    )
    solr_args.add_argument(
        "-cp",
        "--checkpoint-file",
        type=str,
        default=".checkpoint.json",
        help="Path to file used for saving Solr cursor checkpoint.",
    )
    solr_args.add_argument(
        "-rc",
        "--resume-checkpoint",
        action="store_true",
        help="Resume from previously saved checkpoint.",
    )

    developer_args = parser.add_argument_group(
        "Developer Args", "Used for tweaking program's performance."
    )

    developer_args.add_argument(
        "-bs",
        "--buffer-size",
        type=int,
        default=10,
        help="Number of documents to cache while traversing Solr.",
    )

    args = parser.parse_args()

    LOGGING_FOLDER = "./logs"
    LOGGING_FILE = f"{LOGGING_FOLDER}/vector-populator.log"

    logger.auto_configure(args.debug, LOGGING_FILE)
    counts = main(
        args.host,
        args.port,
        args.collection,
        args.request_file,
        args.recieve_file,
        args.checkpoint_file,
        args.resume_checkpoint,
        args.query,
        args.sort,
        args.buffer_size,
    )

    logger.info(f"Total updated: {counts[0]}. Total Failed: {counts[1]}")
