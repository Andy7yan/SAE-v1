# import time
# from datasets import load_dataset

# DATASET_NAME = "allenai/dolma"
# SPLIT = "train"
# RUN_SECOND = 2.0
# PREVIEW_CHARS = 200

# def extract(example: dict) -> str:
#     preferred_keys = ["text", "content", "document", "body"]

#     for key in preferred_keys:
#         value = example.get(key)
#         if isinstance(value, str) and value.strip():
#             return value.strip()

#     for _, value in example.items():
#         if isinstance(value, str) and value.strip():
#             return value.strip()
        
#     return ""

# def main():
#     start_time = time.time()
#     num_examples = 0

#     ds = load_dataset(DATASET_NAME, split=SPLIT, streaming=True)
#     for example in ds:
#         elapsed = time.time() - start_time
#         if elapsed > RUN_SECOND:
#             print("\n Time limit reached, stopping.")
#             break

#         text = extract(example)
#         if not text:
#             continue

#         num_examples += 1
#         preview = text[:PREVIEW_CHARS].replace("\n", " ")
#         print(f"\n example {num_examples} ---")
#         print(f" {preview} ...")

#     total_time = time.time() - start_time
#     print(f"\nProcessed {num_examples} examples in {total_time:.2f} seconds.")

# if __name__ == "__main__":
#     main()

import gzip
import json
import time
import urllib.request
from typing import Optional

# A public Dolma raw shard URL listed in the repository's URL manifest.
# This tests the real data path, not the legacy Hugging Face dataset script.
DOLMA_URL = "https://olmo-data.org/dolma-v1_6/books/books-0000.json.gz"

RUN_SECONDS = 2.0
PREVIEW_CHARS = 200
USER_AGENT = "Mozilla/5.0"


def extract_text(row: dict) -> str:
    value = row.get("text", "")
    if isinstance(value, str):
        return value.strip()
    return ""


def safe_preview(text: str, max_chars: int = PREVIEW_CHARS) -> str:
    return text.replace("\n", " ")[:max_chars]


def open_remote_gzip_stream(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    response = urllib.request.urlopen(request, timeout=30)
    return gzip.GzipFile(fileobj=response)


def main() -> None:
    print(f"Opening remote Dolma shard:\n{DOLMA_URL}\n")
    start = time.time()
    count = 0

    try:
        with open_remote_gzip_stream(DOLMA_URL) as gz_file:
            for raw_line in gz_file:
                elapsed = time.time() - start
                if elapsed >= RUN_SECONDS:
                    print("\nTime limit reached. Stopping.")
                    break

                try:
                    line = raw_line.decode("utf-8")
                    row = json.loads(line)
                except Exception as e:
                    print(f"[skip] Failed to parse one line: {e}")
                    continue

                text = extract_text(row)
                if not text:
                    continue

                count += 1
                print(f"\n--- example {count} ---")
                print(safe_preview(text))

    except Exception as e:
        print(f"\nFAILED: could not stream Dolma raw data.")
        print(f"Error type: {type(e).__name__}")
        print(f"Error: {e}")
        return

    total = time.time() - start
    print(f"\nFinished. Printed {count} examples in {total:.2f} seconds.")


if __name__ == "__main__":
    main()