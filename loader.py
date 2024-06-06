"""
数据加载
"""

import json
import re
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from transformers import BertTokenizer

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = load_vocab(config['pretrain_model_path'])
        self.schema = self.load_schema(config["schema_path"])
        self.sentences = []
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = [9] # cls_token
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                    attention_mask = [1] * len(sentence)
                sentence = ''.join(sentence)
                self.sentences.append(sentence)
                input_ids = self.encode_sentence(sentence)
                labels = self.padding(labels, -1)
                attention_mask = self.padding(attention_mask, -1)
                self.data.append([torch.LongTensor(input_ids), 
                                  torch.LongTensor(attention_mask),
                                  torch.LongTensor(labels)])
        return

    def encode_sentence(self, text): ### 文字转化成了Embedding向量
        return self.tokenizer.encode(text,
                                     padding = 'max_length',
                                     max_length = self.config['max_length'],
                                     truncation = True,
                                     return_attention_mask = True)
    
    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

#加载字表或词表
def load_vocab(vocab_path):
    return BertTokenizer.from_pretrained(vocab_path)

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size = config["batch_size"], shuffle = shuffle,collate_fn = my_collate_fn)
    return dl

def my_collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    return input_ids, attention_mask, labels

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("ner_data/train", Config)
    dl = DataLoader(dg, batch_size = 32)
    for x, y, z in dl:
        print(x[1].shape, y[1].shape,z[1].shape)
        print(x[1], y[1], z[1])
        break