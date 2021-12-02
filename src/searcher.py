import json
from re import T
import urllib.request
from bert import embedding

def handle_query():
    query = input("Enter query: ")

    # script queryでdense_vectorを使う場合のクエリ
    # script_query = {
    #     "script_score": {
    #         "query": {"match_all": {}},
    #         "script": {
    #             "source":"cosineSimilarity(params.query_vector, doc['text_vector']) + 1.0",
    #             "params":{"query_vector": query_vector}
    #         }
    #     }
    # }

    es_query = {
        "knn": {
            "field": "text_vector",
            "query_vector": embedding(query),
            "k": 10,
            "num_candidates": 100
        },
        "_source": False,
        "fields": ["text"]
    }

    url = 'http://127.0.0.1:9200/livedoor/_knn_search'
    headers = {
        'Content-Type': 'application/json',
    }

    req = urllib.request.Request(url, json.dumps(es_query).encode(), headers)

    with urllib.request.urlopen(req) as res:
        body = res.read().decode()
        print(body)


def run_query_loop():
    while True:
        try:
            handle_query()
        except KeyboardInterrupt:
            return


if __name__ == '__main__':
    run_query_loop()
