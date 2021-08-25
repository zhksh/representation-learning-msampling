import torch.nn as nn
from transformers import DistilBertModel
import utils
import torch
from utils import *


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


    '''pick the [CLS] token for classification'''
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.last_hidden_state[:,0,:])
        x = self.activation(self.hidden(x))
        x = self.classifier(x)
        # x = self.softmax(x)
        return x


    def update_embeddings(self, newsize):
        self.base_model.resize_token_embeddings(newsize)


    def evaluate(self, data_loader):
        self.eval()
        device = torch.device('cpu')
        model.to(device)
        accuracy_acc = loss_acc = 0
        with tqdm(data_loader, unit="batch") as batch_generator:
            batch_generator.set_description("Evaluation")
            for c, batch in enumerate(batch_generator, 1):
                X = batch[0]
                X_mask = batch[1]
                Y = batch[2]

                with torch.no_grad():
                    output = self(X, attention_mask=X_mask, labels=Y)
                loss_acc += output.loss.item()
                accuracy_acc += utils.batch_accuracy(output.logits, Y, data_loader.batch_size)
                batch_generator.set_postfix(
                    loss=loss_acc/c,
                    accuracy=100. *  accuracy_acc / c,
                    seen=c * data_loader.batch_size,
                    total=len(data_loader)*data_loader.batch_size)

        return accuracy_acc/len(data_loader)
        
if __name__ == '__main__':

    model = DistilBertSentiment(2)
    print(utils.count_parameters(model))
