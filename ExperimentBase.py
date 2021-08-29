
import os

import pandas as pd
import transformers
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import DistilBertTokenizer
from transformers import logging
import torch.nn as nn
import utils
from utils import *
import torch

class ExperimentBase(nn.Module):
    info = {}

    def __init__(self, conf):
        super(ExperimentBase, self).__init__()
        logging.set_verbosity_error()
    
        self.conf = conf
        self.base_model = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')
        self.path = "{}/{}_{}/".format("checkpoints",self.conf.model_name, self.conf.name)
        if not utils.exists(self.path):
            os.mkdir(self.path)


    def save(self):
        torch.save(self, "{}/{}".format(self.path,"model.torch"))
        with open("{}/{}".format(self.path, 'modelcard.txt'), 'w') as f:
            f.write(self.format_info())

    def load_data(self, data):
        data = self.process_data(data)
        train_dataset = TensorDataset(data["train"]["X"], data["train"]["mask"], data["train"]["Y"])
        test_dataset = TensorDataset(data["test"]["X"], data["test"]["mask"], data["test"]["Y"])
        train_loader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = self.conf.batch_size)
        self.test_loader = DataLoader(test_dataset,sampler = RandomSampler(test_dataset),batch_size = self.conf.batch_size)

        self.train_total = len(train_dataset)
        self.test_total = len(test_dataset)

        print("trainingdata size: {}".format(self.train_total))
        print("testdata size: {}".format(self.test_total))

        return train_loader, self.test_loader


    def update_embeddings(self, newsize):
        self.base_model.resize_token_embeddings(newsize)


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def preprocess_sentences(self, sentences):
        ids = []
        masks = []
        for p in sentences:
            encoded = self.tokenizer.encode_plus(
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


    def process_data(self, data, persist=True):
        ids, masks = self.preprocess_sentences(data["X_train"])
        train_sentence_tensor = torch.cat(ids, dim=0)
        train_masks_tensor = torch.cat(masks, dim=0)
        train_labels_tensor = torch.LongTensor(data["Y_train"])

        ids, masks = self.preprocess_sentences(data["X_test"])
        test_sentence_tensor = torch.cat(ids, dim=0)
        test_masks_tensor = torch.cat(masks, dim=0)
        test_labels_tensor = torch.LongTensor(data["Y_test"])
        # datadir = "data/"
        # if not utils.file_exists(datadir + "train_tensor.pth") or self.conf.reload:
        #     print("processing data")
        #
        #     ids, masks = self.preprocess_sentences(data["X_train"])
        #     train_sentence_tensor = torch.cat(ids, dim=0)
        #     train_masks_tensor = torch.cat(masks, dim=0)
        #     train_labels_tensor = torch.tensor(data["Y_train"])
        #
        #     ids, masks = self.preprocess_sentences(data["X_test"])
        #     test_sentence_tensor = torch.cat(ids, dim=0)
        #     test_masks_tensor = torch.cat(masks, dim=0)
        #     test_labels_tensor = torch.tensor(data["Y_test"])
        #
        #     if persist:
        #         torch.save(train_sentence_tensor, datadir + "train_tensor.pth")
        #         torch.save(train_masks_tensor, datadir + "train_masks.pth")
        #         torch.save(train_labels_tensor, datadir + "train_labels.pth")
        #
        #         torch.save(test_sentence_tensor, datadir + "test_tensor.pth")
        #         torch.save(test_masks_tensor, datadir + "test_masks.pth")
        #         torch.save(test_labels_tensor, datadir + "test_labels.pth")
        #
        # else:
        #     print("loading data")
        #     train_sentence_tensor = torch.load( datadir + "train_tensor.pth")
        #     train_masks_tensor = torch.load(datadir + "train_masks.pth")
        #     train_labels_tensor = torch.load(datadir + "train_labels.pth")
        #
        #     test_sentence_tensor = torch.load(datadir + "test_tensor.pth")
        #     test_masks_tensor = torch.load(datadir + "test_masks.pth")
        #     test_labels_tensor = torch.load(datadir + "test_labels.pth")

        self.info['train tensor shape'] = train_sentence_tensor.shape
        self.info['test tensor shape'] = test_labels_tensor.shape
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

    def forward(self):
        raise NotImplementedError("implement forward()")

    def evaluate(self):
        raise NotImplementedError("implementevaluation!" )

    # def batch_accuracy(self):
    #     raise Exception("implement batch_accuarcy !" )

    def get_info(self):
        info = {}
        info['ts'] = utils.get_formatted_ts()
        info['transformers version'] = transformers.__version__
        info ['model'] =  self.conf.model_name
        info ['description'] = self.conf.desc
        info ['Using device'] = str(self.device)
        info['number of trainable params'] = "{}".format(
            self.count_parameters())
        info['config'] = str(self.conf)
        info.update(self.info)
        self.info = info
        
        return self.info


    def format_info(self):
        info = ""
        for d,v in self.get_info().items():
            info += "{} : {}{}".format(d,v, "\n")
        return info

    @staticmethod
    def batch_accuracy(logits, Y, batch_size):
        Y_ = torch.argmax(logits, dim=1)
        return (Y_ == Y).sum().item() / batch_size


