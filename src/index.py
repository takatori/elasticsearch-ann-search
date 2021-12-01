import glob

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

import torch
from transformers import BertJapaneseTokenizer, BertModel
from joblib import Parallel, delayed

model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
bert = BertModel.from_pretrained(model_name)
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
max_length = 256

client = Elasticsearch()
INDEX_NAME = "ann_sample"

def index_batch(doc):
    requests = get_request(doc)
    print(requests)
    bulk(client, requests)

def get_request(doc):
    return [{
        "_op_type": "index",
        "_index": INDEX_NAME,
        "text": doc,
        "text_vector": embedding(doc)
    }]

def embedding(text):

    encoding = tokenizer(
        text,
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


if __name__ == '__main__':

    BATCH_SIZE = 1000
    docs = []
    count = 0

    category_list = [
        'dokujo-tsushin',
        'it-life-hack',
        'kaden-channel',
        'livedorr-homme',
        'movie-enter',
        'peachy',
        'smax',
        'sports-watch',
        'topic-news'
    ]


    for category in category_list:
        for file in glob.glob(f"../data/text/{category}/{category}*"):
            lines = open(file).read().splitlines()
            print(lines[0])
            text = '\n'.join(lines[3:])
            index_batch(text)
