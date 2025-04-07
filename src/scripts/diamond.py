import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import click

@click.command()
@click.option('--ont', '-ont')

def main(ont):
  train = pd.read_csv(f'../base/{ont}_train.csv')
  val = pd.read_csv(f'../base/{ont}_val.csv')
  test = pd.read_csv(f'../base/{ont}_test.csv')
  ontologies_names = pd.read_csv(f'../base/{ont}_val.csv').columns[2:].values

  labels = train.iloc[:, 2:].values
  dict_ids_train = {train.iloc[:, 0].values[i]: i for i in range(len(train.iloc[:, 0]))}
  nlabels = len(ontologies_names)

  seq_train = preprocess(train)
  seq_val = preprocess(val)
  seq_test = preprocess(test)

  with open(f'../base/reference-{ont}.fasta', 'w') as f:
    print(seq_train, file=f)

  with open(f'../base/queries-{ont}-train.fasta', 'w') as f:
    print(seq_train, file=f)

  with open(f'../base/queries-{ont}-val.fasta', 'w') as f:
    print(seq_val, file=f)

  with open(f'../base/queries-{ont}-test.fasta', 'w') as f:
    print(seq_test, file=f)

  os.system(f'../diamond makedb --in ../base/reference-{ont}.fasta -d ../base/reference-{ont}')

  run_diamond('val', val, ont, nlabels, labels, dict_ids_train)
  run_diamond('test', test, ont, nlabels, labels, dict_ids_train)

  generate_aln('test', ont, test)
  generate_aln('val', ont, val)
  generate_aln('train', ont, train)

def run_diamond(set_name, df, ont, nlabels, labels, dict_ids_train):
  output = os.popen(f'../diamond blastp -d ../base/reference-{ont}.dmnd -q ../base/queries-{ont}-{set_name}.fasta --outfmt 6 qseqid sseqid bitscore nident qlen slen -e 0.001').readlines()

  results = {f'{i}': {'bitscore': [], 'train_id': []} for i in df.iloc[:, 0].values}

  for lines in output:
    line = lines.strip('\n').split()
    results[line[0]]['bitscore'].append(float(line[2]) * (float(line[3]) / max(float(line[4]), float(line[5]))))
    results[line[0]]['train_id'].append(line[1])

  preds = []
  for s in df.iloc[:, 0].values:
    protein_pred = np.zeros(nlabels, dtype=np.float32)
    if len(results[s]['bitscore']) > 0:
      weights = np.array(results[s]['bitscore'])/np.sum(results[s]['bitscore'])
      for i, t_id in enumerate(results[s]['train_id']):
        idx = dict_ids_train[t_id]
        this_pred = labels[idx]
        protein_pred += weights[i] * this_pred
    preds.append(protein_pred)
  preds = np.array(preds)
  np.save(f'../preds/{ont}-diamond-{set_name}.npy', preds)

def generate_aln(set_name, ont, df):
  output = os.popen(f'../diamond blastp -d ../base/reference-{ont}.dmnd -q ../base/queries-{ont}-{set_name}.fasta --outfmt 6 qseqid sseqid bitscore nident qlen slen -e 0.001 --max-target-seqs 5').readlines()

  results = {}

  for lines in output:
    line = lines.strip('\n').split()
    if line[0] != line[1]:
      if line[0] not in results:
        results[line[0]] = {'subject': [], 'bitscore': []}
      results[line[0]]['subject'].append(line[1])
      results[line[0]]['bitscore'].append(float(line[2]) * (float(line[3]) / max(float(line[4]), float(line[5]))))

  final_results = {'query': [], 'subject': [], 'bitscore': []}
  for i in df.iloc[:, 0].values:
    final_results['query'].append(i)
    if i in results:
      final_results['subject'].append(results[i]['subject'])
      final_results['bitscore'].append(results[i]['bitscore'])
    else:
      final_results['subject'].append([])
      final_results['bitscore'].append([])

  pd.DataFrame(final_results).to_csv(f'../base/alignment-{set_name}-{ont}.csv', index=False)

def preprocess(df):
  seq = df.iloc[:, 1].values
  id = df.iloc[:, 0].values
  fasta = ''
  for i, s in enumerate(seq):
    fasta += '>' + str(id[i]) + '\n'
    fasta += s
    fasta += '\n'
  return fasta

if __name__ == '__main__':
  main()
