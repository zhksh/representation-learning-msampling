import torch
import torch.nn as nn
from transformers import DebertaModel, DebertaTokenizer

import utils
from ExperimentBase import ExperimentBase
from tqdm import tqdm


class DeBertaSentiment(ExperimentBase):
    def __init__(self, conf, num_classes, hidden_size = 768, dropout_rate=0.3):
        self.conf = conf
        self.conf.model_name = "deberta-base"
        self.actual_model = "microsoft/deberta-base"
        super(DeBertaSentiment, self).__init__(conf)
        self.base_model = DebertaModel.from_pretrained(self.actual_model)
        self.tokenizer = DebertaTokenizer.from_pretrained(self.actual_model)

        self.num_classes = num_classes

        '''ClassificationHead'''
        self.hidden = nn.Linear(self.base_model.encoder.rel_embeddings.embedding_dim, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.Tanh()




class DeBertaSentimentAvg(DeBertaSentiment):
    def __init__(self, conf, num_classes, hidden_size = 768, dropout_rate=0.3):
        super(DeBertaSentimentAvg, self).__init__(conf, num_classes, hidden_size=hidden_size, dropout_rate=dropout_rate )
        self.conf.model_name += "_avg"



    '''avg all hidden states for classification'''
    def forward(self, input_ids, attention_mask=None):

        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        x = torch.mean(outputs.last_hidden_state, dim=1)
        x = self.dropout(x)
        x = self.activation(self.hidden(x))
        x = self.classifier(x)
        return x



class DeBertaSentimentCLS(DeBertaSentiment):
    def __init__(self, conf, num_classes, hidden_size = 1024, dropout_rate=0.3):
        super(DeBertaSentimentCLS, self).__init__(conf, num_classes, hidden_size=hidden_size, dropout_rate=dropout_rate )
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
    model = DeBertaSentimentCLS(conf, 2)
    print(model.count_parameters())
