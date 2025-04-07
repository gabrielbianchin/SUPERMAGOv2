import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import click
import gc
from utils import *
from evaluate import *

@click.command()
@click.option('--ont', '-ont')
@click.option('--layer', '-l')

def main(ont, layer):
  save_path = f'../models/mlp-{ont}-{layer}.pt'

  train = pd.read_csv(f'../base/{ont}_train.csv')
  val = pd.read_csv(f'../base/{ont}_val.csv')
  test = pd.read_csv(f'../base/{ont}_test.csv')
  ontologies_names = pd.read_csv(f'../base/{ont}_val.csv').columns[2:].values
  ic_ont = pd.read_csv(f'../base/{ont}_ic.csv')
  ic = ic_ont.set_index('terms')['IC'].to_dict()
  output_neurons = len(ontologies_names)

  train_alignments = pd.read_csv(f'../base/alignment-train-{ont}.csv')
  train_alignments['subject'] = train_alignments['subject'].apply(eval)
  train_alignments['bitscore'] = train_alignments['bitscore'].apply(eval)

  val_alignments = pd.read_csv(f'../base/alignment-val-{ont}.csv')
  val_alignments['subject'] = val_alignments['subject'].apply(eval)
  val_alignments['bitscore'] = val_alignments['bitscore'].apply(eval)

  test_alignments = pd.read_csv(f'../base/alignment-test-{ont}.csv')
  test_alignments['subject'] = test_alignments['subject'].apply(eval)
  test_alignments['bitscore'] = test_alignments['bitscore'].apply(eval)

  if ont == 'bp':
    ontology = generate_ontology('../base/go.obo', specific_space=True, name_specific_space='biological_process')
    root = 'GO:0008150'
  elif ont == 'cc':
    ontology = generate_ontology('../base/go.obo', specific_space=True, name_specific_space='cellular_component')
    root = 'GO:0005575'
  else:
    ontology = generate_ontology('../base/go.obo', specific_space=True, name_specific_space='molecular_function')
    root = 'GO:0003674'

  train_dataset = MLPDataset(names_queries=train_alignments.iloc[:, 0].values.tolist(),
                             names_subject=train_alignments.iloc[:, 1].values.tolist(),
                             bitscores=train_alignments.iloc[:, 2].values.tolist(),
                             labels=train.iloc[:, 2:].values,
                             layer=layer)

  val_dataset = MLPDataset(names_queries=val_alignments.iloc[:, 0].values.tolist(),
                             names_subject=val_alignments.iloc[:, 1].values.tolist(),
                             bitscores=val_alignments.iloc[:, 2].values.tolist(),
                             labels=val.iloc[:, 2:].values,
                             layer=layer)

  test_dataset = MLPDataset(names_queries=test_alignments.iloc[:, 0].values.tolist(),
                             names_subject=test_alignments.iloc[:, 1].values.tolist(),
                             bitscores=test_alignments.iloc[:, 2].values.tolist(),
                             labels=test.iloc[:, 2:].values,
                             layer=layer)


  train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
  test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

  if int(layer) > 30:
    shapes = [2560, 2560]
  else:
    shapes = [1024, 1024]

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = MLP(num_outputs=output_neurons, shapes=shapes).to(device)

  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
  criterion = ProteinLoss(weight_tensor=ic_ont.IC.values)

  num_epochs = 50
  best_eval = float('-inf')
  patience = 5
  patience_counter = 0

  for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_dataloader):
      optimizer.zero_grad()
      features = batch["features"]
      features = [feat.to(device) for feat in features]
      labels = batch["label"].to(device).float()
      outputs = model(features)
      loss = criterion(outputs, labels)
      total_loss += loss.item()
      loss.backward()
      optimizer.step()

    avg_loss = total_loss / len(train_dataloader)

    model.eval()
    val_loss = 0
    y_pred_val = []
    y_true_val = []
    with torch.no_grad():
      for val_batch in tqdm(val_dataloader):
        features = val_batch["features"]
        features = [feat.to(device) for feat in features]
        labels = val_batch["label"].to(device).float()
        outputs = model(features)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        preds_val = torch.sigmoid(outputs)
        y_pred_val.append(preds_val.cpu().numpy())
        y_true_val.append(val_batch["label"].cpu().numpy())

    avg_val_loss = val_loss / len(val_dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}, Validation Loss: {avg_val_loss}')

    f_pred, f_true = [], []
    for i in y_pred_val:
      for j in i:
        f_pred.append(j)
    for i in y_true_val:
      for j in i:
        f_true.append(j)
    f_pred = np.array(f_pred)
    f_true = np.array(f_true)
    evaluations = evaluate(f_pred, f_true, ontologies_names, ontology, ic, root)

    if evaluations['fmax'] > best_eval:
      best_eval = evaluations['fmax']
      patience_counter = 0
      torch.save(model.state_dict(), save_path)
      print('Saving model...')
    else:
      patience_counter += 1
      print(f'Patience: {patience_counter}/{patience}')

    if patience_counter >= patience:
      print('Early stopping')
      break

  print('Loading best model...')
  model = MLP(num_outputs=output_neurons, shapes=shapes)
  model.load_state_dict(torch.load(save_path, weights_only=True))
  model.to(device)
  model.eval()

  predict_set(val_dataloader, model, device, "val", ont, layer)
  predict_set(test_dataloader, model, device, "test", ont, layer)

  del model
  gc.collect()
  torch.cuda.empty_cache()

def predict_set(set_dataloader, model, device, set_name, ont, layer):
  y_pred = []
  with torch.no_grad():
    for batch in tqdm(set_dataloader):
      features = batch["features"]
      features = [feat.to(device) for feat in features]
      labels = batch["label"].to(device)
      outputs = model(features)
      pred = torch.sigmoid(outputs)
      y_pred.append(pred.cpu().numpy())

  f_pred = []
  for i in y_pred:
    for j in i:
      f_pred.append(j)
  f_pred = np.array(f_pred)
  np.save(f'../preds/{ont}-mlp-{layer}-{set_name}.npy', f_pred)

if __name__ == '__main__':
  main()
