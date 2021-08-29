from transformers import BertForSequenceClassification, BertTokenizer

from ExperimentBase import ExperimentBase
import utils
import torch
from utils import *
from tqdm import tqdm




class BertSentimentWithHead(ExperimentBase):
    def __init__(self, conf):
        self.conf = conf
        self.conf.model_name = "bert-base-uncased"
        super(BertSentimentWithHead, self).__init__(conf)
        self.tokenizer = BertTokenizer.from_pretrained(self.conf.model_name)
        self.base_model = BertForSequenceClassification.from_pretrained(self.conf.model_name, num_labels=5)


    def forward(self, *args, **kwargs ):
        return self.base_model(*args, **kwargs)


    def evaluate(self, data_loader, device=None):
        self.eval()
        if device is None:
            device = self.device
        # device = torch.device('cpu')
        self.to(device)
        accuracy_acc = loss_acc = 0
        losses = []
        with tqdm(data_loader, unit="batch") as batch_generator:
            batch_generator.set_description("Evaluation")
            for c, batch in enumerate(batch_generator, 1):
                X = batch[0].to(device)
                X_mask = batch[1].to(device)
                Y = batch[2].to(device)

            with torch.no_grad():
                output = self(X, attention_mask=X_mask, labels=Y)
                loss_acc += output.loss.item()
                losses_batch_avg = loss_acc / c
                losses.append(losses_batch_avg)
                accuracy_acc += self.batch_accuracy(output.logits, Y, data_loader.batch_size)
                batch_generator.set_postfix(
                    loss=losses_batch_avg,
                    accuracy=100. *  accuracy_acc / c,
                    seen=c * data_loader.batch_size,
                    total=len(data_loader)*data_loader.batch_size)

        return accuracy_acc/len(data_loader), losses


if __name__ == '__main__':
    conf = utils.read_conf()
    model = BertSentimentWithHead(conf)
    print(model.count_parameters())
