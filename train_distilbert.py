#!/usr/bin/python
import argparse

import pandas as pd
import transformers
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import DistilBertTokenizer
from transformers import logging

import utils
from DistilBertSentiment import DistilBertSentiment
from utils import *

logging.set_verbosity_error()

parser = argparse.ArgumentParser()
parser.add_argument("train_file", default="data/train.tsv", type=str)
parser.add_argument("--reload", default=False, type=bool)
parser.add_argument("--model_name", default="distilbert-base-uncased", type=str)
parser.add_argument("--num_epochs", default=5, type=int )
parser.add_argument("--learning_rate", default=0.00001, type=float)
parser.add_argument("--split", default=0.1, type=float)
parser.add_argument("--batch_size", default=16, type=int)

conf = parser.parse_args()

print(conf)


if __name__ == '__main__':
    print('transformers version :', transformers.__version__)

    tokenizer = DistilBertTokenizer.from_pretrained(conf.model_name)
    tokenizer.add_tokens(["[CLS]"])

    model = DistilBertSentiment(5)
    model.update_embeddings(len(tokenizer))

    data = pd.read_csv(conf.train_file, delimiter='\t', usecols = ['Phrase', 'Sentiment'])


    data = utils.prc_data(data.Phrase.values, data.Sentiment.values, tokenizer, split=conf.split, reload=conf.reload)

    train_dataset = TensorDataset(data["train"]["X"], data["train"]["mask"], data["train"]["Y"])
    test_dataset = TensorDataset(data["test"]["X"], data["test"]["mask"], data["test"]["Y"])
    train_loader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = conf.batch_size)
    test_loader = DataLoader(test_dataset,sampler = RandomSampler(test_dataset),batch_size = conf.batch_size)


    train_total = len(train_dataset)
    test_total = len(test_dataset)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Using device: " + str(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

# print(model)

    best_epoch_acc = 0
    print("number of trainable params: {}".format(utils.count_parameters(model)))
    print("starting training")
    for epoch in range(conf.num_epochs):
        model.to(device)
        model.train()

        accuracy_acc = loss_acc = 0
        with tqdm(train_loader, unit="batch") as batch_generator:
            batch_generator.set_description("Epoch {} ".format(epoch))
            for c, batch in enumerate(batch_generator, 1):
                X = batch[0].to(device)
                X_mask = batch[1].to(device)
                Y = batch[2].to(device)
                optimizer.zero_grad()

                output = model(X, attention_mask=X_mask)
                loss = criterion(output, Y)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                loss.backward()
                optimizer.step()

                accuracy_acc += utils.batch_accuracy(output, Y, train_loader.batch_size)
                loss_acc += loss.item()
                batch_generator.set_postfix(
                    loss=loss.item()/c,
                    accuracy=100. *  accuracy_acc / c,
                    seen=c * conf.batch_size,
                    total=train_total)
                pass

        test_accuracy = model.evaluate(test_loader, criterion)
        if test_accuracy > best_epoch_acc:
            torch.save(model, "{}/{}_{}".format("checkpoints", conf.model_name, utils.format_ts(time.time())))
            best_epoch_acc = test_accuracy




