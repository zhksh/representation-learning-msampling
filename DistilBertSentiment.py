import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
import utils
import torch
from utils import *
from ExperimentBase import ExperimentBase
from tqdm import tqdm



class DistilBertSentiment(ExperimentBase):
    def __init__(self, conf, num_classes, hidden_size = 768, dropout_rate=0.3):
        self.conf = conf
        self.conf.model_name = "distilbert-base-uncased"
        super(DistilBertSentiment, self).__init__(conf)
        self.base_model = DistilBertModel.from_pretrained(self.conf.model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.conf.model_name)

        self.num_classes = num_classes

        '''ClassificationHead'''
        '''output dims from last layer of distillbert are known from hugginface'''
        self.hidden = nn.Linear(768, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)


    def evaluate(self, data_loader, criterion=None, device=None):
        self.eval()
        if device is None:
            device = self.device
        # device = torch.device('cpu')
        self.to(device)
        if criterion is None:
            criterion = self.criterion
        accuracy_acc = loss_acc = 0
        losses = []
        accuracies = []
        with tqdm(data_loader, unit="batch") as batch_generator:
            batch_generator.set_description("Evaluation")
            for c, batch in enumerate(batch_generator, 1):
                X = batch[0].to(device)
                X_mask = batch[1].to(device)
                Y = batch[2].to(device)

                with torch.no_grad():
                    output = self(X, attention_mask=X_mask)
                loss = criterion(output, Y)
                loss_acc += loss.item()
                loss_batch_avg = loss_acc /c
                losses.append(loss_batch_avg)
                accuracy_acc += self.batch_accuracy(output, Y, data_loader.batch_size)
                accuracy_avg = accuracy_acc /c
                accuracies.append(accuracy_avg)
                batch_generator.set_postfix(
                    loss=loss_batch_avg,
                    accuracy=100. *  accuracy_avg,
                    seen=c * data_loader.batch_size,
                    total=len(data_loader)*data_loader.batch_size)

        return accuracies, losses



class DistilBertSentimentAvg(DistilBertSentiment):
    def __init__(self, conf, num_classes, hidden_size = 768, dropout_rate=0.3):
        super(DistilBertSentimentAvg, self).__init__(conf, num_classes, hidden_size=hidden_size, dropout_rate=dropout_rate )


    '''avg all hidden states for classification'''
    def forward(self, input_ids, attention_mask=None):

        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        x = torch.mean(outputs.last_hidden_state, dim=1)
        x = self.dropout(x)
        x = self.activation(self.hidden(x))
        x = self.classifier(x)
        return x



class DistilBertSentimentCLS(DistilBertSentiment):
    def __init__(self, conf, num_classes, hidden_size = 768, dropout_rate=0.3):
        super(DistilBertSentimentCLS, self).__init__(conf, num_classes, hidden_size=hidden_size, dropout_rate=dropout_rate )


    '''pick the first token for classification'''
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.last_hidden_state[:,0,:])
        x = self.activation(self.hidden(x))
        x = self.classifier(x)
        return x



if __name__ == '__main__':
    conf = utils.read_conf()
    model = DistilBertSentiment(conf, 2)
    print(model.count_parameters())
