#!/bin/bash
#sample="down"
#name="cls_cls_data_down"
#mode="cls"
#epochs=4
#data="data/train_cls.tsv"
#cross_eval="data/test_referencecls.tsv"
#./train_distilbert.py $data --sample $sample --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "bert with no cls annotated data" &> distilbert_log_$name
##./train_bert.py $data --sample $sample --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "bert with no cls annotated data" &> bert_log_$name
#./train_deberta.py $data --sample $sample --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "deberta with no cls annotated data" &> deberta_log_$name
#
#sample="middle"
#name="cls_nocls_data"
#mode="cls"
#epochs=4
#data="data/train.tsv"
#cross_eval="data/test_reference.tsv"
#./train_bert.py $data --sample $sample --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "bert with no cls annotated data" &> bert_log_$name
#./train_deberta.py $data --sample $sample --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "deberta with no cls annotated data" &> deberta_log_$name
#./train_distilbert.py $data --sample $sample --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "bert with no cls annotated data" &> distilbert_log_$name
#
#
#sample="middle"
#name="cls_cls_data"
#mode="cls"
#epochs=4
#data="data/train_cls.tsv"
#cross_eval="data/test_referencecls.tsv"
#./train_bert.py $data --sample $sample --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "bert with no cls annotated data" &> bert_log_$name
#./train_deberta.py $data --sample $sample --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "deberta with no cls annotated data" &> deberta_log_$name
#./train_distilbert.py $data --sample $sample --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "bert with no cls annotated data" &> distilbert_log_$name
#
#sample="middle"
#name="avg_middle_sample"
#mode="avg"
#epochs=4
#data="data/train.tsv"
#cross_eval="data/test_reference.tsv"
#./train_bert.py $data --sample $sample --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "bert with no cls annotated data" &> bert_log_$name
#./train_deberta.py $data --sample $sample --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "deberta with no cls annotated data" &> deberta_log_$name
#./train_distilbert.py $data --sample $sample --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "bert with no cls annotated data" &> distilbert_log_$name



sample="middle"
name="avg_middle"
mode="avg"
epochs=4
data="data/train.tsv"
cross_eval="data/test_reference.tsv"
./train_deberta.py $data --sample $sample --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "deberta with no cls annotated data" &> deberta_log_$name
#./train_distilbert.py $data --sample $sample --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "bert with no cls annotated data" &> distilbert_log_$name


sample="up"
name="cls_nocls_up"
mode="cls"
epochs=4
data="data/train.tsv"
cross_eval="data/test_reference.tsv"
./train_bert.py $data --sample $sample --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "bert with no cls annotated data" &> bert_log_$name
./train_deberta.py $data --sample $sample --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "deberta with no cls annotated data" &> deberta_log_$name
./train_distilbert.py $data --sample $sample --name $name --class_mode $mode --cross_eval_file $cross_eval --num_epochs $epochs --desc "bert with no cls annotated data" &> distilbert_log_$name