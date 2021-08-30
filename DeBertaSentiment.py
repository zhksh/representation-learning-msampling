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
        self.softmax = nn.LogSoftmax(dim=1)


    def evaluate(self, data_loader, criterion, device=None):
        self.eval()
        if device is None:
            device = self.device
        # device = torch.device('cpu')
        self.to(device)
        if criterion is None:
            criterion = self.criterion
        accuracy_acc = loss_acc = 0
        losses = []
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
                loss_batch_avg = loss_acc / c
                losses.append(loss_batch_avg)
                accuracy_acc += self.batch_accuracy(output, Y, data_loader.batch_size)
                batch_generator.set_postfix(
                    loss=loss_batch_avg,
                    accuracy=100. *  accuracy_acc / c,
                    seen=c * data_loader.batch_size,
                    total=len(data_loader)*data_loader.batch_size)

        return accuracy_acc/len(data_loader), losses



class DeBertaSentimentAvg(DeBertaSentiment):
    def __init__(self, conf, num_classes, hidden_size = 768, dropout_rate=0.3):
        super(DeBertaSentimentAvg, self).__init__(conf, num_classes, hidden_size=hidden_size, dropout_rate=dropout_rate )
        self.conf.name = "avg"


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
        self.conf.name = "mean"

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
