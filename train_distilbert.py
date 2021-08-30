#!/usr/bin/python

import utils
from DistilBertSentiment import DistilBertSentimentAvg, DistilBertSentimentCLS
from utils import *
import torch
from tqdm import tqdm


if __name__ == '__main__':
    conf = utils.read_conf()
    model = DistilBertSentimentCLS(conf, 5)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    model.criterion = criterion

    best_epoch_acc = 0

    print("starting training")
    for epoch in range(conf.num_epochs):
        model.to(model.device)
        model.train()

        with tqdm(train_loader, unit="batch") as batch_generator:
            batch_generator.set_description("Epoch {} ".format(epoch))
            for c, batch in enumerate(batch_generator, 1):
                X = batch[0].to(model.device)
                X_mask = batch[1].to(model.device)
                Y = batch[2].to(model.device)
                optimizer.zero_grad()

                output = model(X, attention_mask=X_mask)
                loss = criterion(output, Y)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                loss.backward()
                optimizer.step()

                accuracy = model.batch_accuracy(output, Y, train_loader.batch_size)
                avg_acc, avg_loss = model.update_stats("train", accuracy, loss.item())

                batch_generator.set_postfix(
                    loss=avg_loss,
                    accuracy=100. *  avg_acc,
                    seen=c * conf.batch_size,
                    total=model.train_total)

        #per epoch
        epoch_test_accuracy = model.evaluate(test_loader, criterion)
        model.plot_epoch_stats(epoch)


        if  epoch_test_accuracy > best_epoch_acc:
            model.save()
            best_epoch_acc = epoch_test_accuracy




