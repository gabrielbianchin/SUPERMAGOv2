from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer, BitsAndBytesConfig
import torch
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
import click

@click.command()
@click.option('--model_name', '-m')
@click.option('--ont', '-ont')

def main(model_name, ont):

  train, pos_train, names_train = preprocess(df=pd.read_csv(f'../base/{ont}_train.csv'))
  val, pos_val, names_val = preprocess(df=pd.read_csv(f'../base/{ont}_val.csv'))
  test, pos_test, names_test = preprocess(df=pd.read_csv(f'../base/{ont}_test.csv'))
  device = torch.device("cuda:0")

  if model_name == 'esm':
    model_path = 'facebook/esm2_t36_3B_UR50D'
    embed_size = 2560
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, output_hidden_states=True).to(device)

  elif model_name == 't5':
    model_path = 'Rostlab/prot_t5_xl_half_uniref50-enc'
    embed_size = 1024
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5EncoderModel.from_pretrained(model_path, output_hidden_states=True).to(device)

  model.eval()
  process_data(train, pos_train, names_train, tokenizer, model, device, embed_size, ont, model_name)
  process_data(val, pos_val, names_val, tokenizer, model, device, embed_size, ont, model_name)
  process_data(test, pos_test, names_test, tokenizer, model, device, embed_size, ont, model_name)

  del model
  gc.collect()
  torch.cuda.empty_cache()

def process_data(data, pos, names, tokenizer, model, device, embed_size, ont, model_name):
  l1 = np.zeros((len(data), embed_size))
  l2 = np.zeros((len(data), embed_size))
  l3 = np.zeros((len(data), embed_size))

  for i, this_seq in enumerate(tqdm(data)):
    l1[i], l2[i], l3[i] = get_embeddings(this_seq, tokenizer, model, device)
    gc.collect()
    torch.cuda.empty_cache()

  nl1 = protein_embedding(l1, pos)
  nl2 = protein_embedding(l2, pos)
  nl3 = protein_embedding(l3, pos)

  if model_name == 'esm':
    l1_name, l2_name, l3_name = '36', '35', '34'
  else:
    l1_name, l2_name, l3_name = '24', '23', '22'

  for i in range(len(names)):
    # saving embs
    np.save(f'../embs/{names[i]}-{l1_name}.npy', nl1[i])
    np.save(f'../embs/{names[i]}-{l2_name}.npy', nl2[i])
    np.save(f'../embs/{names[i]}-{l3_name}.npy', nl3[i])

    # save imgs
    nl1[i] = sigmoid(nl1[i])
    nl2[i] = sigmoid(nl2[i])
    nl3[i] = sigmoid(nl3[i])
    np.save(f'../embs/img-{names[i]}-{l1_name}.npy', nl1[i])
    np.save(f'../embs/img-{names[i]}-{l2_name}.npy', nl2[i])
    np.save(f'../embs/img-{names[i]}-{l3_name}.npy', nl3[i])

  del l1, l2, l3, nl1, nl2, nl3
  gc.collect()
  torch.cuda.empty_cache()

def sigmoid(x):
  x = 1 / (1 + np.exp(-x))
  x = np.round(x * 255)
  x = x.astype(int)
  return x

def emb_method(matrix_embs, method):
  if method == 'mean':
    return np.mean(matrix_embs, axis=0)

def protein_embedding(X, pos, method='mean'):
  n_X = []
  last_pos = pos[0]
  cur_emb = []
  for i in range(len(X)):
    cur_pos = pos[i]
    if last_pos == cur_pos:
      cur_emb.append(X[i])
    else:
      n_X.append(emb_method(np.array(cur_emb), method))
      last_pos = cur_pos
      cur_emb = [X[i]]
  n_X.append(emb_method(np.array(cur_emb), method))

  return np.array(n_X)

def preprocess(df, subseq=1022):
  prot_list = []
  positions = []
  sequences = df.iloc[:, 1].values
  names = df.iloc[:, 0].values.tolist()
  for i in range(len(sequences)):
    len_seq = int(np.ceil(len(sequences[i]) / subseq))
    for idx in range(len_seq):
      positions.append(i)
      if idx != len_seq - 1:
        prot_list.append(sequences[i][idx * subseq : (idx + 1) * subseq])
      else:
        prot_list.append(sequences[i][idx * subseq :])

  return prot_list, positions, names

def get_embeddings(seq, tokenizer, model, device):
  batch_seq = [" ".join(list(seq))]
  ids = tokenizer(batch_seq)
  input_ids = torch.tensor(ids['input_ids']).to(device)
  attention_mask = torch.tensor(ids['attention_mask']).to(device)

  with torch.no_grad():
    embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

  return embedding_repr.hidden_states[-1][0].detach().cpu().numpy().mean(axis=0), \
         embedding_repr.hidden_states[-2][0].detach().cpu().numpy().mean(axis=0), \
         embedding_repr.hidden_states[-3][0].detach().cpu().numpy().mean(axis=0)

if __name__ == '__main__':
  main()
