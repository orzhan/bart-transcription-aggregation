Code for paper Fine-tuning pre-trained language model for crowdsourced texts aggregation (VLDB 2021 Crowd Science Challenge)

## Introduction

VLDB 2021 Crowd Science Challenge is a shared task on aggregation of crowdsourced texts. Multiple transcriptions made by people needed to be aggregated to produce a single high-quality transcription. The audios were produced using a voice assistant from Wikipedia articles. 

The problem is that some annotators can be unskilled or malicious. One more thing, different people can make mistakes in different parts of the sentence. The data is very noisy. 

The metric used to evaluate the solutions in the shared task was highest Average Word Accuracy (AWAcc). Word Accuracy is calculated as

WAcc = 100 · max(1-WER, 0)

This aggregation task can be seen as a particular case of multi-document summarization or as mistake correction. Pre-trained language models are widely used for many text-related tasks, including text summarization. Linguistic knowledge is beneficial in this task because it helps to choose the possible word sequences, or replace a misheard word with a word with high probability in the context. We applied end-to-end training because the available dataset was large enough.

## Install

`python -r requirements.txt`

## Train

Put `responses.csv` and `gt.csv` from the shared task to the repository root folder. Run `python training.py`

## Evaluate

Best model weights can be downloaded from https://drive.google.com/drive/folders/11h4bqvXHTPXWHZDRBOMaEkWALQ-yCuG4?usp=sharing

Put `test.csv` from the shared task to the repository root folder and run `python inference.py`, it will save predictions to `bart-multi-out.csv`

