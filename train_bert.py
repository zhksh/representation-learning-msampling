#!/usr/bin/python

import utils
from BertSentimentWithHead import BertSentimentWithHead
from utils import *
import torch
from tqdm import tqdm
if __name__ == '__main__':
    conf = utils.read_conf()
    model = BertSentimentWithHead(conf)

    data = pd.read_csv(conf.train_file, delimiter='\t', usecols = ['Phrase', 'Sentiment'])
    plt = utils.show_dist_plot(data["Sentiment"], "Class distribution")
    plt.savefig(model.path +"classdist.png")

    if conf.sample != "None":
        conf.reload = True
        data = utils.sample_data(data, "Sentiment", conf.sample)


    split_data = utils.data_split(
        data.Phrase.values, data.Sentiment.values, model.path,
        split=model.conf.split)

    train_loader, test_loader = model.load_data(split_data)


    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)
    best_epoch_acc = 0
    train_losses = []
    test_losses = []
    print("starting training")
    for epoch in range(conf.num_epochs):
        model.to(model.device)
        model.train()

        accuracy_acc = loss_acc = 0
        with tqdm(train_loader, unit="batch") as batch_generator:
            batch_generator.set_description("Epoch {} ".format(epoch))
            for c, batch in enumerate(batch_generator, 1):
                X = batch[0].to(model.device)
                X_mask = batch[1].to(model.device)
                Y = batch[2].to(model.device)
                optimizer.zero_grad()

                output = model(X, token_type_ids=None,attention_mask=X_mask,labels=Y)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                output.loss.backward()
                optimizer.step()
                loss = output.loss

                accuracy_acc += model.batch_accuracy(output.logits, Y, train_loader.batch_size)
                loss_acc += loss.item()
                batch_generator.set_postfix(
                    loss=loss.item()/c,
                    accuracy=100. *  accuracy_acc / c,
                    seen=c * conf.batch_size,
                    total=model.train_total)

        test_accuracy, test_losses_local = model.evaluate(test_loader)
        test_losses.extend(test_losses_local)
        utils.show_loss_plt(train_losses, test_losses, "{}/{}_{}".format(model.path, "loss_curve_", epoch), model.conf.model_name)
        if test_accuracy > best_epoch_acc:
            model.save()
            best_epoch_acc = test_accuracy




