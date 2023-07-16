import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer, RobertaTokenizer, RobertaModel

import pandas as pd

decoder = RobertaModel.from_pretrained('roberta-large', device_map={"":0})
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
prefix_words = 'Please say whether the tweets is actually announcing \
a disaster, be careful some of them are deceitful. Only answer me\
with one token, 0 for not  a disaster and 1 for is disaster. The tweets is: '
prefix_tokenized = tokenizer(prefix_words, padding=True, truncation=True, return_tensors='pt')
prefix_tokens, prefix_mask = prefix_tokenized.input_ids, prefix_tokenized.attention_mask
prefix_embedding = decoder.embeddings(prefix_tokens)

torch.save(prefix_embedding, '/home/broca/kg/prefix_embedding.pt')
torch.save(prefix_mask, '/home/broca/kg/mask.pt')
# file_path = '/home/broca/kg/train.csv'
# train_file = pd.read_csv(file_path)
# valid_pd = train_file[7400: ]
# train_pd = train_file[: 7400]
# train_pd.reset_index()
# valid_pd.reset_index()
# valid_pd.to_csv('/home/broca/kg/data/valid.csv')
# train_pd.to_csv('/home/broca/kg/data/train.csv')