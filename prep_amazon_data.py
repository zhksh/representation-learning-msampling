#!/usr/bin/python
import pandas as pd

import utils
from utils import *
import json
conf = utils.read_conf()


if __name__ == "__main__":
    json_data = []
    with open(conf.train_file, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            item = json.loads(line)
            try:
                if item['style']['Format:'] == ' Audio CD': continue
                item['overall'] -= 1
                json_data.append({'Sentiment' :item['overall'], 'Phrase': item['reviewText']})
            except Exception as e:
                print(line)
                print(e)
    data = pd.DataFrame(json_data)
    del json_data
    sampled = utils.sample_data(data, "Sentiment", conf.sample)

    plt = utils.show_dist_plot(sampled.Sentiment.values, "Class distribution")
    plt.savefig("{}{}.png".format(conf.train_file, "_classdist"))
    sampled.to_csv('data/test_reference.tsv', sep = '\t')