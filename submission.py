#!/usr/bin/python
import argparse
import pandas as pd
import torch
import transformers
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, BertTokenizer, logging

import utils
from utils import *

logging.set_verbosity_error()

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("data")
parser.add_argument("--model_name", default="bert-base-uncased", type=str)
parser.add_argument("--batch_size", default=100, type=int)



conf = parser.parse_args()

print(conf)

if __name__ == '__main__':
    print('transformers version :', transformers.__version__)

    tokenizer = BertTokenizer.from_pretrained(conf.model_name)
    data = pd.read_csv(conf.data, delimiter='\t', usecols = ['PhraseId', 'Phrase'])
    ids, masks = preprocess_sentences(data.Phrase, tokenizer)
    ids = torch.cat(ids, dim=0)
    masks = torch.cat(masks, dim=0)
    dataset = TensorDataset(ids, masks)
    data_loader = DataLoader(dataset, sampler = SequentialSampler(dataset), batch_size = 20)


    total = len(dataset)
    model = torch.load(conf.model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device: " + str(device))

    print("starting submission creation")
    model.to(device)
    model.eval()
    predictions = []
    with tqdm(data_loader, unit="batch") as batch_generator:
        batch_generator.set_description("Submission")
        for c, batch in enumerate(batch_generator, 1):
            X = batch[0].to(device)
            X_mask = batch[1].to(device)

            output = model(X,token_type_ids=None,attention_mask=X_mask)
            Y =  torch.argmax(output.logits, dim=1)
            predictions.extend((Y.tolist()))

    df_submission = pd.DataFrame(list(zip(data.PhraseId.values, predictions)), columns=['PhraseId', 'Sentiment'])
    df_submission.to_csv("submission.tsv", index=False)






