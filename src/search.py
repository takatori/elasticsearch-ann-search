import json
from re import T

from elasticsearch import Elasticsearch
import urllib.request

import torch
from transformers import BertJapaneseTokenizer, BertModel
from joblib import Parallel, delayed

model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
bert = BertModel.from_pretrained(model_name)
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)


client = Elasticsearch()
max_length = 256

def embedding(query):
    encoding = tokenizer(
        query,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    attention_mask = encoding['attention_mask']

    # 文章ベクトルを計算
    # BERTの最終層の出力の平均を計算する。(ただし、[PAD]は除く)
    with torch.no_grad():
        output = bert(**encoding)
        last_hidden_state = output.last_hidden_state
        averaged_hidden_state = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) \
            / attention_mask.sum(1, keepdim=True)

    return averaged_hidden_state[0].tolist()    

def handle_query(): 
    query = input("Enter query: ")
    query_vector = embedding(query)
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
            "query_vector": query_vector,
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
    # print(req.data)

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