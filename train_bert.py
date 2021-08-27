#!/usr/bin/python

import utils
from BertSentimentWithHead import BertSentimentWithHead
from utils import *

if __name__ == '__main__':
    conf = utils.read_conf()
    model = BertSentimentWithHead(conf)
    data = pd.read_csv(conf.train_file, delimiter='\t', usecols = ['Phrase', 'Sentiment'])

    if conf.sample != "None":
        conf.reload = True
        data = utils.sample_data(data, "Sentiment", conf.sample)

    data = utils.prc_data(data.Phrase.values, data.Sentiment.values, model.tokenizer, split=conf.split, reload=conf.reload)
    train_loader, test_loader = model.load_data(data)


    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)
    best_epoch_acc = 0
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

        test_accuracy = model.evaluate(test_loader)
        if test_accuracy > best_epoch_acc:
            torch.save(model, "{}/{}_{}".format("checkpoints", conf.model_name, utils.format_ts(time.time())))
            best_epoch_acc = test_accuracy



