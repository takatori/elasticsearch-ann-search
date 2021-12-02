import glob
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from bert import embedding
from joblib import Parallel, delayed


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
