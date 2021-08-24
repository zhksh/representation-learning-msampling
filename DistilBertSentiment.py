import torch.nn as nn
from transformers import DistilBertModel
import utils
import torch


class DistilBertSentiment(nn.Module):
    def __init__(self, num_classes, hidden_size = 768, dropout_rate=0.3):
        super(DistilBertSentiment, self).__init__()
        self.num_classes = num_classes
        self.base_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

        self.hidden = nn.Linear(768, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.last_hidden_state[:,0,:])
        x = self.activation(self.hidden(x))
        x = self.classifier(x)
        # x = self.softmax(x)
        return x


if __name__ == '__main__':

    model = DistilBertSentiment(2)
    print(utils.count_parameters(model))
