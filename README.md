# Sentiment Classification on Movie Reviews (Kaggle)
This is the final project for the Neural Representation Seminar SS21 @LMU 

https://www.kaggle.com/c/movie-review-sentiment-analysis-kernels-only

## Model
``bert-base-uncased`` from the transformers lib from huggingface


## Usage
```bash
./train.py path/to/training_data --split .1
```
will read data file and splits it in trainig and testet by 0.1

```bash
./eval.py path/to/checkpoint/model path/to/data/testdata
```
will evaluate a saved model

```bash
./submission.py path/to/checkpoint/model path/to/data/testdata
```
will create an output file in the format required by the kaggle competition 

## Results
testset accuracy is generally slightly above 70%