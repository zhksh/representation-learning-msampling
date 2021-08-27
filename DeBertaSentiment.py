import torch.nn as nn
from transformers import DebertaModel, DebertaTokenizer
import utils
import torch
from utils import *
from ExperimentBase import ExperimentBase


class DeBertaSentiment(ExperimentBase):
    def __init__(self, conf, num_classes, hidden_size = 768, dropout_rate=0.3):
        super(DeBertaSentiment, self).__init__(conf)
        self.conf.model_name = "microsoft/deberta-base"
        self.base_model = DebertaModel.from_pretrained(self.conf.model_name)
        self.tokenizer = DebertaTokenizer.from_pretrained(self.conf.model_name)

        self.num_classes = num_classes

        '''ClassificationHead'''
        self.hidden = nn.Linear(self.base_model.encoder.rel_embeddings.embedding_dim, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)

        self.print_info()


    def evaluate(self, data_loader, criterion):
        self.eval()
        device = torch.device('cpu')
        self.to(device)
        accuracy_acc = loss_acc = 0
        with tqdm(data_loader, unit="batch") as batch_generator:
            batch_generator.set_description("Evaluation")
            for c, batch in enumerate(batch_generator, 1):
                X = batch[0]
                X_mask = batch[1]
                Y = batch[2]

                with torch.no_grad():
                    output = self(X, attention_mask=X_mask)
                loss = criterion(output, Y)
                loss_acc += loss.item()
                accuracy_acc += self.batch_accuracy(output, Y, data_loader.batch_size)
                batch_generator.set_postfix(
                    loss=loss_acc/c,
                    accuracy=100. *  accuracy_acc / c,
                    seen=c * data_loader.batch_size,
                    total=len(data_loader)*data_loader.batch_size)

        return accuracy_acc/len(data_loader)



class DeBertaSentimentAvg(DeBertaSentiment):
    def __init__(self, conf, num_classes, hidden_size = 768, dropout_rate=0.3):
        super(DeBertaSentimentAvg, self).__init__(conf, num_classes, hidden_size=hidden_size, dropout_rate=dropout_rate )


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
