# -*- coding: utf-8 -*-
#!pip install simpletransformers jellyfish jiwer wandb

#from google.colab import drive
#drive.mount('/content/drive')

import pandas as pd
import numpy as np
import random
import json


def filter_text(text):
    char_set = set(list(' qwertyuiopasdfghjklzxcvbnm\''))
    return ''.join([c for c in text.strip().lower() if c in char_set])

df=pd.read_csv('responses.csv')
gt=pd.read_csv('gt.csv')

"""## Seq2seq"""

import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs

from jiwer import wer
from tqdm import tqdm

import random
random.seed(32)

# Augmented train by permutations

train_size=9400

train=[]
eval=[]
test=[]
for i in range(df.task.min(), df.task.max()+1):
  seqs = [x.replace('.','').strip().lower() for x in df[df.task==i]['output'].values]
  s0 = '|'.join(seqs)
  if i < train_size:
    for k in range(0,3):
      random.shuffle(seqs)
      s0 = '|'.join(seqs)
      train.append([s0, gt[gt.task==i].iloc[0]['output']])
  elif i < 9700:
    eval.append([s0, gt[gt.task==i].iloc[0]['output']])
  else:
    test.append(s0)

random.shuffle(train)
train=pd.DataFrame(train)
train.columns = ["input_text", "target_text"]
eval=pd.DataFrame(eval)
if len(eval) > 0:
  eval.columns = ["input_text", "target_text"]

# Configure the model
model_args = Seq2SeqArgs()
model_args.num_train_epochs = 5
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.evaluate_during_training_steps = 1000
model_args.wandb_project = 'toloka_bart'
model_args.learning_rate = 4e-6
model_args.gradient_accumulation_steps = 1
model_args.overwrite_output_dir = True
model_args.num_beams = 5
model_args.use_multiprocessing = False

model_args.max_seq_length = 160
model_args.max_length = 60
model_args.length_penalty = 1.0

model_args.train_batch_size=8
model_args.eval_batch_size=8

model_args.save_steps = -1
model_args.save_eval_checkpoints = True

model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="facebook/bart-large",
    args=model_args,
)

def wer_eval(pred, true):
  wacc = []
  j= 0
  for t in true:
      wacc.append(100 * max(1 - wer(pred[j], t), 0))
      j+=1
  return sum(wacc) / len(wacc)

# Train the model
model.train_model(train, eval_data=eval, wer=wer_eval)



eval_preds = model.predict([x.replace('.','').strip().lower() for x in eval.input_text.values.tolist()])

wacc = []
j= 0
for i, row in eval.iterrows():
    wacc.append(100 * max(1 - wer(eval_preds[i], row['target_text']), 0))
    j+=1
print('Evaluation score', sum(wacc) / len(wacc))

test=[]
test_df=pd.read_csv('test.csv')
for i in range(test_df.task.min(), test_df.task.max()+1):
  seqs = [x.replace('.','').strip().lower() for x in test_df[test_df.task==i]['output'].values]
  s0 = '|'.join(seqs)
  test.append(s0)

preds = model.predict(test)

out=[]
j=0
for i in range(test_df.task.min(), test_df.task.max()+1):
  out.append({'task':i, 'output': preds[j]})
  j+=1
pd.DataFrame(out).to_csv('bart-out.csv', index=False)

print('Errors on eval dataset')
for i,r in eval.iterrows():
  if r['target_text'] != eval_preds[i]:
    print(i, r['target_text'])
    print(eval_preds[i])
    print('-'*50)

