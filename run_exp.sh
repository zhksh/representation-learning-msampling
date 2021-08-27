#!/bin/bash

./train_bert.py data/train.tsv --sample up --name sampleup_cls --num_epochs 10 --desc "bert with upsampled no cls" &> bert_log
./train_deberta.py data/train.tsv --sample up --name sampleup_cls --num_epochs 10 --desc "bert with upsampled no cls" &> deberta_log
./train_distilbert.py data/train.tsv --sample up --name sampleup_cls --num_epochs 10 --desc "bert with upsampled no cls" &> distilbert_log