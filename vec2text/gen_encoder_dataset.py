# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("prajjwal1/bert-medium")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-medium")

import os
import datasets
training_data_path = '/home/ramvenkat98/.cache/inversion/0aaa9cff054220b8af32ddcf5a1e837b.arrow'
train_dataset = datasets.load_from_disk(training_data_path)['train']

import transformers
t5_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-base', padding = True, truncation = 'max_length', max_length = 32)

import torch

def generate_x_and_y(initial_dataset):
    y = initial_dataset['frozen_embeddings']
    x_initial = initial_dataset['embedder_input_ids']
    decoded_x_initial = t5_tokenizer.batch_decode(x_initial[:, :-1])
    d = {}
    for s in decoded_x_initial:
        l = len(s.split())
        if l not in d:
            d[l] = 1
        else:
            d[l] += 1
    print(d)
    x_arguments = tokenizer(decoded_x_initial, padding = True, max_length = 50, truncation = True, return_attention_mask = True, return_tensors = 'pt')
    return x_arguments['input_ids'], x_arguments['attention_mask'], y

input_ids, attention_mask, y = generate_x_and_y(train_dataset)

torch.save({'input_ids': input_ids, 'attention_mask': attention_mask, 'y': y}, 'encoder_train.pth')
