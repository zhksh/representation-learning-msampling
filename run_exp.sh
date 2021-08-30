#!/bin/bash

sample="middle"
name="cls_nocls_data"
mode="cls"
epochs=7
data="data/train.tsv"
cross_eval="data/test_reference.tsv"
./train_bert.py $data --sample $middle --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "bert with no cls annotated data" &> bert_log_$name
./train_deberta.py $data --sample $middle --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "deberta with no cls annotated data" &> deberta_log_$name
./train_distilbert.py $data --sample $middle --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "bert with no cls annotated data" &> distilbert_log_$name


sample="middle"
name="cls_cls_data"
mode="cls"
epochs=7
data="data/train_cls.tsv"
cross_eval="data/test_reference_cls.tsv"
./train_bert.py $data --sample $middle --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "bert with no cls annotated data" &> bert_log_$name
./train_deberta.py $data --sample $middle --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "deberta with no cls annotated data" &> deberta_log_$name
./train_distilbert.py $data --sample $middle --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "bert with no cls annotated data" &> distilbert_log_$name

sample="middle"
name="avg_middle_sample"
mode="avg"
epochs=7
data="data/train.tsv"
cross_eval="data/test_reference.tsv"
./train_bert.py $data --sample $middle --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "bert with no cls annotated data" &> bert_log_$name
./train_deberta.py $data --sample $middle --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "deberta with no cls annotated data" &> deberta_log_$name
./train_distilbert.py $data --sample $middle --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "bert with no cls annotated data" &> distilbert_log_$name

sample="down"
name="cls_cls_data_down"
mode="cls"
epochs=7
data="data/train_cls.tsv"
cross_eval="data/test_reference_cls.tsv"
./train_bert.py $data --sample $middle --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "bert with no cls annotated data" &> bert_log_$name
./train_deberta.py $data --sample $middle --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "deberta with no cls annotated data" &> deberta_log_$name
./train_distilbert.py $data --sample $middle --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "bert with no cls annotated data" &> distilbert_log_$name