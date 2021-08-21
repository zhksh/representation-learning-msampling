#!/usr/bin/python
import argparse
import pandas as pd
import transformers
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, BertTokenizer, logging



import utils
from utils import *

logging.set_verbosity_error()

parser = argparse.ArgumentParser()
parser.add_argument("--train_file", default="data/train.tsv", type=str)
parser.add_argument("--reload", default=False, type=bool)
parser.add_argument("--model_name", default="bert-base-uncased", type=str)
parser.add_argument("--split", default=0.1, type=float)

conf = parser.parse_args()

print(conf)


if __name__ == '__main__':
    print('transformers version :', transformers.__version__)

    tokenizer = BertTokenizer.from_pretrained(conf.model_name)
    data_frame = pd.read_csv(conf.train_file, delimiter='\t', usecols = ['Phrase', 'Sentiment'])


    data = utils.prc_data(data_frame.Phrase.values, data.Sentiment.values, tokenizer, split=conf.split, reload=conf.reload)

    train_dataset = TensorDataset(data["train"]["X"], data["train"]["mask"], )
    test_dataset = TensorDataset(data["test"]["X"], data["test"]["mask"], data["test"]["Y"])
    train_loader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = conf.batch_size)
    test_loader = DataLoader(test_dataset,sampler = RandomSampler(test_dataset),batch_size = conf.batch_size)


    labels = data["train"]["Y"].tolist()
    estimator =  lambda x: len(x) / len(labels) * 100
    show_barplot(labels, title="training set label dist", estimator=estimator)
    plt.savefig('img/train_class_dist.png')
    labels = data["test"]["Y"].tolist()
    estimator =  lambda x: len(x) / len(labels) * 100
    show_barplot(labels, title="testset label dist", estimator=estimator)
    plt.savefig('img/test_class_dist.png')




