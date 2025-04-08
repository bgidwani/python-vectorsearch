
from VectorSearch import VectorSearch
import time

searcher = VectorSearch()

def __time_elapsed_for_function(fn, *args):
    start_time = time.time()
    result = fn(*args)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Load time: {elapsed_time} seconds")
    return result


def load_data():
    __time_elapsed_for_function(searcher.ingest_file, "data.csv")


def save_embeddings_data():
    __time_elapsed_for_function(searcher.save_embeddings, "embeddings.json")

def load_embeddings_data():
    __time_elapsed_for_function(searcher.load_embeddings, "embeddings.json")


def validate_search(query: str):
    start_time = time.time()

    results = searcher.search(query, top_k=2)

    end_time = time.time()
    print(f"Results for {query} in {end_time - start_time} seconds")

    for score, text in results:
        print(f" == Score: {score:.4f}, Text: {text}")

    print('=' * 20)


load_data()
save_embeddings_data()
load_embeddings_data()

items = ["citrus",
         "apple tart"]

for item in items:
    validate_search(item)