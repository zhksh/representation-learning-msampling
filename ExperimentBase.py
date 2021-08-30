
import os

import torch
import torch.nn as nn
import transformers
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import tqdm
from transformers import logging

import utils


class ExperimentBase(nn.Module):
    info = {}

    def __init__(self, conf):
        super(ExperimentBase, self).__init__()
        logging.set_verbosity_error()

        self.conf = conf
        self.base_model = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.path = "{}/{}_{}/".format("checkpoints",self.conf.model_name, self.conf.name)
        if not utils.exists(self.path):
            os.mkdir(self.path)
        self.stats = ExperimentBase.make_stats_dict()

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


    def add_token(self, token):
        self.tokenizer.add_tokens([token])
        self.update_embeddings(len(self.tokenizer))


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
                max_length = self.conf.max_length,
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


    def evaluate(self, data_loader, criterion=None, device=None):
        self.eval()
        if device is None:
            device = self.device
        # device = torch.device('cpu')
        self.to(device)
        if criterion is None:
            criterion = self.criterion
        avg_acc = 0
        with tqdm(data_loader, unit="batch") as batch_generator:
            batch_generator.set_description("Evaluation")
            for c, batch in enumerate(batch_generator, 1):
                X = batch[0].to(device)
                X_mask = batch[1].to(device)
                Y = batch[2].to(device)

                with torch.no_grad():
                    output = self(X, attention_mask=X_mask)
                loss = criterion(output, Y)
                accuracy = self.batch_accuracy(output, Y, data_loader.batch_size)
                avg_acc, avg_loss = self.update_stats("test", accuracy, loss.item())

                batch_generator.set_postfix(
                    loss=avg_loss,
                    accuracy=100. *  avg_acc,
                    seen=c * data_loader.batch_size,
                    total=len(data_loader)*data_loader.batch_size)

        return avg_acc


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


    def update_stats(self,mode, acc, loss):
        self.stats[mode]["accuracies"].append(acc)
        self.stats[mode]["losses"].append(loss)
        normalized_acc = sum(self.stats[mode]["accuracies"])/len(self.stats[mode]["accuracies"])
        normalized_loss = sum(self.stats[mode]["losses"])/len(self.stats[mode]["losses"])
        self.stats[mode]["normalized_accuracies"].append(normalized_acc)
        self.stats[mode]["normalized_losses"].append(normalized_loss)

        return normalized_acc, normalized_loss


    def plot_epoch_stats(self, epoch):
        data = self.stats
        utils.show_loss_plt(data["train"]["normalized_losses"], data["test"]["normalized_losses"], "{}/{}_{}".format(
            self.path, "loss_curve_", epoch),
                            "{} epoch {}".format(
                                self.conf.model_name , epoch))
        utils.show_acc_plt(data["train"]["normalized_accuracies"], data["test"]["normalized_accuracies"], "{}/{}_{}".format(
            self.path, "accuracy_curve_", epoch),
                       "{} epoch {}".format(
                           self.conf.model_name , epoch))






    @staticmethod
    def batch_accuracy(logits, Y, batch_size):
        Y_ = torch.argmax(logits, dim=1)
        return (Y_ == Y).sum().item() / batch_size


    @staticmethod
    def make_stats_dict():
        return {
            "train" : {
                "accuracies" : [],
                "losses" : [],
                "normalized_accuracies" : [],
                "normalized_losses" : []
            },
            "test" : {
                "accuracies" : [],
                "losses" : [],
                "normalized_accuracies" : [],
                "normalized_losses" : []
            }
        }
