#!/usr/bin/python
import argparse

import pandas as pd
import transformers
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer, logging

import utils
from utils import *
from sklearn.utils import resample

logging.set_verbosity_error()

parser = argparse.ArgumentParser()
parser.add_argument("train_file", default="data/train.tsv", type=str)
parser.add_argument("--reload", default=False, type=bool)
parser.add_argument("--model_name", default="bert-base-uncased", type=str)
parser.add_argument("--num_epochs", default=5, type=int )
parser.add_argument("--learning_rate", default=0.00001, type=float)
parser.add_argument("--split", default=0.1, type=float)
parser.add_argument("--batch_size", default=16, type=int)

conf = parser.parse_args()

print(conf)


if __name__ == '__main__':
    print('transformers version :', transformers.__version__)

    tokenizer = BertTokenizer.from_pretrained(conf.model_name)
    data = pd.read_csv(conf.train_file, delimiter='\t', usecols = ['Phrase', 'Sentiment'])
    data = utils.upsample_data(data, "Sentiment")

    data = utils.prc_data(data.Phrase.values, data.Sentiment.values, tokenizer, split=conf.split, reload=True)


    train_dataset = TensorDataset(data["train"]["X"], data["train"]["mask"], data["train"]["Y"])
    test_dataset = TensorDataset(data["test"]["X"], data["test"]["mask"], data["test"]["Y"])
    train_loader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = conf.batch_size)
    test_loader = DataLoader(test_dataset,sampler = RandomSampler(test_dataset),batch_size = conf.batch_size)


    train_total = len(train_dataset)
    test_total = len(test_dataset)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
