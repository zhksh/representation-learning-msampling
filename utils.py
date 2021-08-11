from tqdm import tqdm
import time
import torch
from os.path import exists
from sklearn.model_selection import train_test_split


def preprocess_sentences(sentences, tokenizer):
    ids = []
    masks = []
    for p in sentences:
        encoded = tokenizer.encode_plus(
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


def file_exists(filename):
    return exists(filename)

def prc_data(X, Y, tokenizer, name="", split=.0, reload=False):
    datadir = "data/"
    if not file_exists(datadir + "train_tensor.pth") or reload:
        print("processing data")
        train_data, test_data, train_labels, test_labels = train_test_split(
            X, Y, test_size=split, stratify=Y)

        ids, masks = preprocess_sentences(train_data, tokenizer)
        train_sentence_tensor = torch.cat(ids, dim=0)
        train_masks_tensor = torch.cat(masks, dim=0)
        train_labels_tensor = torch.tensor(train_labels)

        ids, masks = preprocess_sentences(test_data, tokenizer)
        test_sentence_tensor = torch.cat(ids, dim=0)
        test_masks_tensor = torch.cat(masks, dim=0)
        test_labels_tensor = torch.tensor(test_labels)

        torch.save(train_sentence_tensor, datadir + "train_tensor.pth")
        torch.save(train_masks_tensor, datadir + "train_masks.pth")
        torch.save(train_labels_tensor, datadir + "train_labels.pth")

        torch.save(test_sentence_tensor, datadir + "test_tensor.pth")
        torch.save(test_masks_tensor, datadir + "test_masks.pth")
        torch.save(test_labels_tensor, datadir + "test_labels.pth")

    else:
        print("loading data")
        train_sentence_tensor = torch.load( datadir + "train_tensor.pth")
        train_masks_tensor = torch.load(datadir + "train_masks.pth")
        train_labels_tensor = torch.load(datadir + "train_labels.pth")

        test_sentence_tensor = torch.load(datadir + "test_tensor.pth")
        test_masks_tensor = torch.load(datadir + "test_masks.pth")
        test_labels_tensor = torch.load(datadir + "test_labels.pth")

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


def eval(model, data_loader):
    model.eval()
    device = torch.device('cpu')
    model.to(device)
    accuracy_acc = 0
    total_eval_loss = 0
    c = 0
    with tqdm(data_loader, unit="batch") as batch_generator:
        batch_generator.set_description("Evaluation")
        for c, batch in enumerate(batch_generator, 1):
            X = batch[0]
            X_mask = batch[1]
            Y = batch[2]

            with torch.no_grad():
                output = model(X, token_type_ids=None, attention_mask=X_mask,labels=Y)
            total_eval_loss += output.loss.item()
            accuracy_acc += batch_accuracy(output.logits, Y, data_loader.batch_size)
            batch_generator.set_postfix(
                loss=total_eval_loss/c,
                accuracy=100. *  accuracy_acc / c,
                seen=c * data_loader.batch_size,
                total=len(data_loader)*data_loader.batch_size)

    return accuracy_acc/len(data_loader)

def batch_accuracy(logits, Y, batch_size):
    Y_ = torch.argmax(logits, dim=1)
    return (Y_ == Y).sum().item() / batch_size


def format_ts(ts):
    return time.ctime(ts).replace(" ", "_")