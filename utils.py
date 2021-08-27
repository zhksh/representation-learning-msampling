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

import utils



def file_exists(filename):
    return exists(filename)


def data_split(X, Y, path, split=0.1):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split, stratify=Y)
    train_df = pd.DataFrame({"train": Y_train})
    test_df = pd.DataFrame({"test": Y_test})
    plot_df = pd.DataFrame({"train": train_df.value_counts().values,
                            "test": test_df.value_counts().values},
                           index=test_df.value_counts().index)
    ax = plot_df.plot.bar(rot=0)
    plt.savefig(path + "data_dist.png")
    return {"X_train" : X_train,
            "X_test" : X_test,
            "Y_train" : Y_train,
            "Y_test" : Y_test}

def format_ts(ts):
    return time.ctime(ts).replace(" ", "_")


def get_formatted_ts():
    return format_ts(time.time())


def show_dist_plot(data, title, estimator=None):
    plt.clf()
    if estimator is None:
        estimator =  lambda x: len(x) / len(data) * 100
    ax = sns.barplot(x=data, y=data,  estimator=estimator)
    ax.set(ylabel="Percent")
    ax.set(xlabel="Class")
    # sns.countplot(data["train"]["Y"].tolist())
    plt.title("{} (total {})".format(title, len(data)))
    # plt.show()
    return plt

def show_loss_plt(train_losses, test_losses, path, name):
    plt.clf()
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss ({})".format(name))
    plt.plot(test_losses,label="test")
    plt.plot(train_losses,label="train")
    plt.xlabel("#samples")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path + ".png")
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
    parser.add_argument("--model_name", default="nomodelnamegiven", type=str)
    parser.add_argument("--num_epochs", default=5, type=int )
    parser.add_argument("--learning_rate", default=0.00001, type=float)
    parser.add_argument("--split", default=0.1, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--sample", default="None", choices=['down', 'up'], type=str)
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--desc", default="", type=str)

    return parser.parse_args()