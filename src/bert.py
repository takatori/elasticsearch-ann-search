import torch
from transformers import BertJapaneseTokenizer, BertModel

model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
bert = BertModel.from_pretrained(model_name)
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)


def embedding(text):

    encoding = tokenizer(
        text,
        max_length=256,
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
