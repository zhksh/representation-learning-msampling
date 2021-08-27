from tqdm import tqdm
import time
import torch
from os.path import exists
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
import pandas as pd
import argparse


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


def file_exists(filename):
    return exists(filename)

def prc_data(X, Y, tokenizer, split=.0, reload=False, persist=True, prefix = ""):
    datadir = "data/" + prefix + "_"
    if not file_exists(datadir + "train_tensor.pth") or reload:
        print("processing data")
        train_data, test_data, train_labels, test_labels = train_test_split(
            X, Y, test_size=split, stratify=Y)

        ids, masks = preprocess_sentences(train_data, tokenizer)
        train_sentence_tensor = torch.cat(ids, dim=0)
        train_masks_tensor = torch.cat(masks, dim=0)
        train_labels_tensor = torch.tensor(train_labels)

        ids, masks = preprocess_sentences(test_data, tokenizer)
        test_sentence_tensor = torch.cat(ids, dim=0)
        test_masks_tensor = torch.cat(masks, dim=0)
        test_labels_tensor = torch.tensor(test_labels)

        if persist:
            torch.save(train_sentence_tensor, datadir + "train_tensor.pth")
            torch.save(train_masks_tensor, datadir + "train_masks.pth")
            torch.save(train_labels_tensor, datadir + "train_labels.pth")

            torch.save(test_sentence_tensor, datadir + "test_tensor.pth")
            torch.save(test_masks_tensor, datadir + "test_masks.pth")
            torch.save(test_labels_tensor, datadir + "test_labels.pth")

    else:
        print("loading data")
        train_sentence_tensor = torch.load( datadir + "train_tensor.pth")
        train_masks_tensor = torch.load(datadir + "train_masks.pth")
        train_labels_tensor = torch.load(datadir + "train_labels.pth")

        test_sentence_tensor = torch.load(datadir + "test_tensor.pth")
        test_masks_tensor = torch.load(datadir + "test_masks.pth")
        test_labels_tensor = torch.load(datadir + "test_labels.pth")

    return {
            "train" : {
                "X": train_sentence_tensor,
                "mask" : train_masks_tensor,
                "Y" : train_labels_tensor
            },
            "test" :{
                "X" : test_sentence_tensor,
                "mask": test_masks_tensor,
                "Y" : test_labels_tensor
            }
    }




def format_ts(ts):
    return time.ctime(ts).replace(" ", "_")


def get_formatted_ts():
    return format_ts(time.time())

def show_barplot(data, title, estimator=None):
    ax = sns.barplot(x=data, y=data,  estimator=estimator)
    ax.set(ylabel="Percent")
    ax.set(xlabel="Class")
    # sns.countplot(data["train"]["Y"].tolist())
    plt.title("{} (total {})".format(title, len(data)))
    # plt.show()





def sample_data(data, col_name, mode):
    class_dist = data[col_name].value_counts()
    print("class distribution before {}sampling".format(mode))
    print(class_dist)
    bound =  class_dist.values.max() if mode == "up" else class_dist.values.min()
    sampled_classes = []
    for c, count in class_dist.items():
        ups = resample(data[data["Sentiment"] == c],
                       replace=True,
                       n_samples=bound,
                       random_state=42)
        sampled_classes.append(ups)
    balanced = pd.concat(sampled_classes)
    print("after")
    print(balanced[col_name].value_counts())
    return balanced


def read_conf():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file", default="data/train.tsv", type=str)
    parser.add_argument("--reload", default=False, type=bool)
    parser.add_argument("--model_name", default="distilbert-base-uncased", type=str)
    parser.add_argument("--num_epochs", default=5, type=int )
    parser.add_argument("--learning_rate", default=0.00001, type=float)
    parser.add_argument("--split", default=0.1, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--sample", default="None", choices=['down', 'up'], type=str)
    parser.add_argument("--name", default="", type=str)

    return parser.parse_args()