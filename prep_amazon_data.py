#!/usr/bin/python
import pandas as pd

import utils
from utils import *
import json
conf = utils.read_conf()


if __name__ == "__main__":
    data = None
    if conf.reload or not utils.exists("data/test_reference_full.tsv"):
        json_data = []
        with open(conf.train_file, 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                item = json.loads(line)
                try:
                    if 'reviewText' not in item : continue
                    item['overall'] -= 1
                    json_data.append({'Sentiment' :item['overall'], 'Phrase': item['reviewText']})
                except Exception as e:
                    print(line)
                    print(e)
        data = pd.DataFrame(json_data)
        del json_data
        plt = utils.show_dist_plot(data.Sentiment.values, "Class distribution before " +conf.sample+"sampling")
        plt.savefig("{}{}.png".format(conf.train_file, "_classdist_before"))
    else:
        data = pd.read_csv("data/test_reference_full.tsv", delimiter='\t', usecols = ['Phrase', 'Sentiment'])


    sampled = utils.sample_data(data, "Sentiment", conf.sample)

    plt = utils.show_dist_plot(sampled.Sentiment.values, "Class distribution after " +conf.sample+"sampling")
    plt.savefig("{}{}.png".format(conf.train_file, "_classdist_after"))
    sampled.to_csv('data/test_reference_full.tsv', sep = '\t')

    split = utils.data_split(sampled.Phrase.values, sampled.Seniment.values, "data/reference")
    # this will give as a stratified subset, the original set is really large
    reduced_data = pd.DataFrame({'Sentiment': split['Y_test'], 'Phrase': split['X_test']})
    plt = utils.show_dist_plot(sampled.Sentiment.values, "Class distribution of subset after " +conf.sample+"sampling")
    plt.savefig("{}{}.png".format(conf.train_file, "_subset_classdist_after"))
    reduced_data.to_csv('data/test_reference.tsv', sep = '\t')