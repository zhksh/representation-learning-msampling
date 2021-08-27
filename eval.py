#!/usr/bin/python
import argparse

import pandas as pd
import transformers
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer, logging

import utils
from utils import *

logging.set_verbosity_error()

parser = argparse.ArgumentParser()
parser.add_argument("--loader_path", type=str)
parser.add_argument("path", type=str)

conf = parser.parse_args()

print(conf)


if __name__ == '__main__':
    # tokenizer = BertTokenizer.from_pretrained(conf.model_name)
    # data_eval = pd.read_csv(conf.eval_file, delimiter='\t',usecols = ['Phrase','Sentiment'])
    # ids, masks = preprocess_sentences(data_eval.Phrase.values, tokenizer)
    # X = torch.cat(ids, dim=0)
    # X_mask = torch.cat(masks, dim=0)
    # Y = torch.tensor(data_eval.Sentiment.values)
    # test_dataset = TensorDataset(X, X_mask, Y)
    # test_loader = DataLoader(test_dataset, sampler = RandomSampler(test_dataset), batch_size = 1)

    model = torch.load(conf.path + "/model.torch")
    accuracy = model.evaluate(model.test_loader, torch.nn.CrossEntropyLoss())






