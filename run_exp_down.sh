#!/bin/bash

./train_bert.py data/train.tsv --sample down --max_length 150--name sampledown --num_epochs 10 --desc "bert with downsampled no cls" &> bert_log__down
./train_deberta.py data/train.tsv --sample down --max_length 150--name sampledown --num_epochs 10 --desc "bert with downsampled no cls" &> deberta_log_down
./train_distilbert.py data/train.tsv --sample down --max_length 150--name sampledown --num_epochs 10 --desc "bert with downsampled no cls" &> distilbert_log_down


#./train_bert.py data/train.tsv --sample up --name sampleup_cls --num_epochs 10 --desc "bert with upsampled no cls" &> bert_log
#./train_deberta.py data/train.tsv --sample up --name sampleup_cls --num_epochs 10 --desc "bert with upsampled no cls" &> deberta_log
#./train_distilbert.py data/train.tsv --sample up --name sampleup_cls --num_epochs 10 --desc "bert with upsampled no cls" &> distilbert_log