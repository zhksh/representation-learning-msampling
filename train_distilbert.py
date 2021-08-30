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
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
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

                output = model(X, attention_mask=X_mask)
                loss = criterion(output, Y)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                loss.backward()
                optimizer.step()
                accuracy = model.batch_accuracy(output, Y, train_loader.batch_size)
                accuracy_acc += accuracy
                accuracy_acc_avg = accuracy_acc / c
                train_accuracies.append(accuracy_acc_avg)
                loss_acc += loss.item()
                loss_batch_avg = loss_acc / c
                train_losses.append(loss_batch_avg)
                batch_generator.set_postfix(
                    loss=loss_batch_avg,
                    accuracy=100. *  accuracy_acc_avg,
                    seen=c * conf.batch_size,
                    total=model.train_total)

        #per epoch
        test_accuracies, test_losses_local = model.evaluate(test_loader, criterion)
        test_accuracies.extend(test_accuracies)
        test_losses.extend(test_losses_local)
        utils.show_loss_plt(train_losses, test_losses, "{}/{}_{}".format(
            model.path, "loss_curve_", epoch),
                            "{} epoch {}".format(
                                model.conf.model_name , epoch))
        utils.show_acc_plt(train_accuracies, test_accuracies, "{}/{}_{}".format(
            model.path, "accuracy_curve_", epoch),
                            "{} epoch {}".format(
                                model.conf.model_name , epoch))
        epoch_test_accuarcy = sum(test_accuracies)/len(test_accuracies)
        if  epoch_test_accuarcy > best_epoch_acc:
            model.save()
            best_epoch_acc = epoch_test_accuarcy




