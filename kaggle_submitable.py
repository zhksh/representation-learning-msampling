#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import torch
import transformers
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer, logging


# for TPU
#import torch_xla
#import torch_xla.core.xla_model as xm


logging.set_verbosity_error()

class Object(object):
    pass

conf = Object()

conf.num_epochs = 5
conf.learning_rate = .00001
conf.batch_size = 8
conf.model_name = "bert-base-uncased"



def preprocess_sentences(sentences, tokenizer):
    ids = []
    masks = []
    for p in sentences:
        encoded = tokenizer.encode_plus(
            p,
            truncation=True,
            add_special_tokens = True,
            max_length = 80, ##todo
            padding='max_length',
            pad_to_max_length = True,
            return_attention_mask = True,
            return_tensors = 'pt'
        )
        ids.append(encoded['input_ids'])
        masks.append(encoded['attention_mask'])
    return ids, masks


def prc_data(X, Y, tokenizer):
    print("processing data")
    ids, masks = preprocess_sentences(X, tokenizer)
    train_sentence_tensor = torch.cat(ids, dim=0)
    train_masks_tensor = torch.cat(masks, dim=0)
    train_labels_tensor = torch.tensor(Y)

    return {
        "X": train_sentence_tensor,
        "mask" : train_masks_tensor,
        "Y" : train_labels_tensor
    }


def batch_accuracy(logits, Y, batch_size):
    Y_ = torch.argmax(logits, dim=1)
    return (Y_ == Y).sum().item() / batch_size



if __name__ == '__main__':
    print('transformers version :', transformers.__version__)

    data = pd.read_csv("/kaggle/input/movie-review-sentiment/train.tsv/train.tsv", delimiter='\t', usecols = ['Phrase', 'Sentiment'])
    data = pd.read_csv("data/train.tsv", delimiter='\t', usecols = ['Phrase', 'Sentiment'])

    tokenizer = BertTokenizer.from_pretrained(conf.model_name)
    data = prc_data(data.Phrase.values, data.Sentiment.values, tokenizer)

    train_dataset = TensorDataset(data["X"], data["mask"], data["Y"])
    train_loader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = conf.batch_size)

    train_total = len(train_dataset)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = xm.xla_device()
    #torch.set_default_tensor_type('torch.FloatTensor')
    print("Using device: " + str(device))
    model = BertForSequenceClassification.from_pretrained(conf.model_name, num_labels=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)
    print("starting training")
    for epoch in range(conf.num_epochs):
        model.to(device)
        model.train()

        epoch_loss = 0
        local_loss = 0
        local_acc = 0
        acc_loss = 0
        with tqdm(train_loader, unit="batch") as batch_generator:
            batch_generator.set_description("Epoch {} ".format(epoch))
            for c, batch in enumerate(batch_generator, 1):
                X = batch[0].to(device)
                X_mask = batch[1].to(device)
                Y = batch[2].to(device)
                optimizer.zero_grad()

                output = model(X,token_type_ids=None,attention_mask=X_mask,labels=Y)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                output.loss.backward()
                optimizer.step()
                loss = output.loss

                local_acc += batch_accuracy(output.logits, Y, train_loader.batch_size)
                local_loss += loss.item()
                batch_generator.set_postfix(
                    loss=loss.item()/c,
                    accuracy=100. *  local_acc / c,
                    seen=c * conf.batch_size,
                    total=train_total)


    data_eval = pd.read_csv("data/test.tsv",delimiter='\t',usecols = ['PhraseId','Phrase'])
    ids, masks = preprocess_sentences(data_eval.Phrase, tokenizer)
    ids = torch.cat(ids, dim=0)
    masks = torch.cat(masks, dim=0)
    dataset = TensorDataset(ids, masks)
    data_loader = DataLoader(dataset, sampler = SequentialSampler(dataset), batch_size = 20)

    total = len(dataset)

    print("starting submission creation")
    predictions = []
    with tqdm(data_loader, unit="batch") as batch_generator:
        batch_generator.set_description("Submission")
        for c, batch in enumerate(batch_generator, 1):
            X = batch[0].to(device)
            X_mask = batch[1].to(device)

            output = model(X,token_type_ids=None,attention_mask=X_mask)
            Y =  torch.argmax(output.logits, dim=1)
            predictions.extend((Y.tolist()))

    df_submission = pd.DataFrame(list(zip(data_eval.PhraseId.values, predictions)), columns=['PhraseId', 'Sentiment'])
    df_submission.to_csv("submission.tsv", index=False)










