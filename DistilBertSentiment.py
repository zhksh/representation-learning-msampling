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




class DistilBertSentimentAvg(DistilBertSentiment):
    def __init__(self, conf, num_classes, hidden_size = 768, dropout_rate=0.3):
        super(DistilBertSentimentAvg, self).__init__(conf, num_classes, hidden_size=hidden_size, dropout_rate=dropout_rate )
        self.conf.model_name += "_avg"

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
        self.conf.model_name += "_cls"


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
