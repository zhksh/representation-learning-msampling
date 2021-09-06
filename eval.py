#!/usr/bin/python

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import utils
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)

parser.add_argument("eval_file", type=str)
parser.add_argument("--device", default="", type=str)


conf = parser.parse_args()

if __name__ == '__main__':
    model = torch.load(conf.path + "/model.torch")
    if conf.eval_file =="self":
        accuracy = model.evaluate(model.test_loader)
        model.info['test acc'] = accuracy
        print(model.format_info())
        exit(0)

    if conf.device != "":
        model.device = torch.device(conf.device)
        model.to(model.device)

    data_eval = pd.read_csv(conf.eval_file, delimiter='\t',usecols = ['Phrase','Sentiment'])
    ids, masks = model.preprocess_sentences(data_eval.Phrase.values)
    X = torch.cat(ids, dim=0)
    X_mask = torch.cat(masks, dim=0)
    Y = torch.LongTensor(data_eval.Sentiment.values)
    test_dataset = TensorDataset(X, X_mask, Y)
    test_loader = DataLoader(test_dataset, sampler = RandomSampler(test_dataset), batch_size = 16)
    model.reset_stats()
    accuracy = model.evaluate(test_loader)







