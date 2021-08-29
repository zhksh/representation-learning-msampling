#!/usr/bin/python

import utils
from utils import *

conf = utils.read_conf()


if __name__ == "__main__":
    data = pd.read_json(conf.train_file,  lines=True)
    #unnest style struct
    style_col = pd.json_normalize(data["style"])
    data['style'] = style_col
    #make consistent to rating of kaggle data 0-4
    data.overall = data.overall.apply(lambda x : x-1)
    data = data[data['style'] != " Audio CD"]
    target_df = pd.DataFrame({"Sentiment" :data.overall, "Phrase" : data.reviewText})
    sampled = utils.sample_data(target_df, "Sentiment", conf.sample)

    plt = utils.show_dist_plot(sampled.Sentiment.values, "Class distribution")
    plt.savefig("{}{}.png".format(conf.train_file, "_classdist"))
    sampled.to_csv('data/test_reference.tsv', sep = '\t')