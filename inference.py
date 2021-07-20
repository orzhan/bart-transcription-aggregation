import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs

from jiwer import wer
from tqdm import tqdm
import itertools
from itertools import combinations, cycle
import requests
import re
from collections import Counter


def count_swaps(word, cmpr):
  swaps = 0
  chars = {c: [] for c in word}
  [chars[c].append(i) for i, c in enumerate(word)]
  for k in chars.keys():
      chars[k] = cycle(chars[k])
  idxs = [next(chars[c]) for c in cmpr]
  for cmb in combinations(idxs, 2):
      if cmb[0] > cmb[1]:
          swaps += 1
  return swaps
  

url ="https://raw.githubusercontent.com/hyperreality/American-British-English-Translator/master/data/british_spellings.json"
british_to_american_dict = requests.get(url).json()    

def americanize(string):
    for british_spelling, american_spelling in british_to_american_dict.items():
        br = british_spelling
        am = american_spelling
        if british_spelling.find('se') != -1 or british_spelling.find('sing') != -1:
          continue
        string = string.replace(" " + br + " ",  " " + am + " ")
        #string = re.sub(r"\b%s\b" % british_spelling, american_spelling, string)
  
    return string
  
  

# Configure the model
model_args = Seq2SeqArgs()
model_args.num_beams = 5
model_args.use_multiprocessing = False

model_args.max_seq_length = 160
model_args.max_length = 60
model_args.length_penalty = 1.0

model_args.eval_batch_size=32


npred=20

model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="checkpoint-17625-epoch-5",
    args=model_args,
)

permutations=list(itertools.permutations(range(0,7)))

gpp=[permutations[0]]
for i in tqdm(range(1, npred), total=npred-1):
  gmax=0
  gadd=[]
  for j in permutations:
    if j in gpp:
      continue
    gc = 0
    for k in gpp:
      gc += count_swaps(k, j)
    if gc > gmax:
      gmax = gc
      gadd = j
  gpp += [gadd]



test0=[]
test=[]
test_df=pd.read_csv('test.csv')
for i in range(test_df.task.min(), test_df.task.max()+1):
  seqs = [filter_text(x).strip().lower() for x in test_df[test_df.task==i]['output'].values]
  test0.append('|'.join(seqs))
  for p in gpp:
    s0 = '|'.join([seqs[t] for t in p])
    test.append(s0)
	
	
preds0 = model.predict(test0)	
out=[]
j=0
for i in range(test_df.task.min(), test_df.task.max()+1):
  out.append({'task':i, 'output': preds0[j]})
  j+=1
#pd.DataFrame(out).to_csv('bart-single-out.csv', index=False)

preds = model.predict(test)



	
final_preds=[]
for i in range(test_df.task.min(), test_df.task.max()+1):
  j = i-test_df.task.min()
  i_preds = [americanize(x) for x in preds[j * npred:(j+1)*npred] + [preds0[j]]]
  c = Counter()
  for p in i_preds:
    c[p] += 1
  final_preds.append((c.most_common(1)[0])[0])
	
	
out=[]
j=0
for i in range(test_df.task.min(), test_df.task.max()+1):
  out.append({'task':i, 'output': final_preds[j]})
  j+=1
pd.DataFrame(out).to_csv('bart-multi-out.csv', index=False)	
	
	
	
	
	
	
	
	
	
	
	
	
	

