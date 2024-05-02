import sys

import torch

from transformers import T5EncoderModel, T5Tokenizer
from transformers import BertModel, BertTokenizer
from transformers import XLNetModel, XLNetTokenizer
from transformers import AlbertModel, AlbertTokenizer

import gc
model_name = "Rostlab/prot_t5_xl_uniref50"
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if "t5" in model_name:
  tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False )
  model = T5EncoderModel.from_pretrained(model_name)
elif "albert" in model_name:
  tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=False )
  model = AlbertModel.from_pretrained(model_name)
elif "bert" in model_name:
  tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False )
  model = BertModel.from_pretrained(model_name)
elif "xlnet" in model_name:
  tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=False )
  model = XLNetModel.from_pretrained(model_name)
else:
  print("Unkown model name")

gc.collect()
print("Number of model parameters is: " + str(int(sum(p.numel() for p in model.parameters())/1000000)) + " Million")

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
model = model.eval()
if torch.cuda.is_available():
  model = model.half()

def read_sequence(sequence_file):

  sequence_dict = dict()

  f = open(sequence_file, "r")
  text = f.read()
  f.close()

  for line in text.split():
    line = line.strip()
    if (line.startswith(">")):
      name = line[1:]
    else:

      sequence = line
      sequence_dict[name] = sequence

  return sequence_dict

def embed_dataset(seq, shift_left = 0, shift_right = -1):


  with torch.no_grad():
    ids = tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding=True, is_split_into_words=True,
                                      return_tensors="pt")
    embedding = model(input_ids=ids['input_ids'].to(device))[0]
    embedding= embedding[0].detach().cpu().numpy()[shift_left:shift_right]


  return embedding

if "t5" in model_name:
  shift_left = 0
  shift_right = -1
elif "bert" in model_name:
  shift_left = 1
  shift_right = -1
elif "xlnet" in model_name:
  shift_left = 0
  shift_right = -2
elif "albert" in model_name:
  shift_left = 1
  shift_right = -1
else:
  print("Unkown model name")


sequence_file = sys.argv[1]
feature_dir = sys.argv[2]
sequence_dict = read_sequence(sequence_file)
os.system("rm -rf " + feature_dir)
os.makedirs(feature_dir)


for name in sequence_dict:

  seq = sequence_dict[name]

  if(len(seq)>1021):
    seq = seq[0: 1021]

  sample = list(seq)
  embedding = embed_dataset(sample, shift_left, shift_right)
  feature_file = feature_dir + "/" + name + ".pt"
  torch.save(embedding, feature_file)