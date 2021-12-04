import glob
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from bert import embedding
from joblib import Parallel, delayed

client = Elasticsearch()
INDEX_NAME = "ann_sample"

def index_batch(docs):
    requests = [get_request(doc) for doc in docs]
    bulk(client, requests)

def get_request(doc):
    return {
        "_op_type": "index",
        "_index": INDEX_NAME,
        "text": doc,
        "text_vector": embedding(doc)
    }


if __name__ == '__main__':

    BATCH_SIZE = 100
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
            text = '\n'.join(lines[3:])

            docs.append(text)
            count += 1

            if count % BATCH_SIZE == 0:
                index_batch(docs)
                docs = []
                print(f"Indexed {count} documents.")

        if docs:    
            index_batch(docs)
            print(f"Indexed {count} documents.")

