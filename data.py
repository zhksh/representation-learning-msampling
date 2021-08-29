#!/usr/bin/python

import utils
from utils import *


conf = utils.read_conf()


if __name__ == "__main__":
    data = pd.read_json(conf.train_file, lines=True, usecols = ['Phrase', 'Sentiment'])
    plt = utils.show_dist_plot(data["Sentiment"], "Class distribution")