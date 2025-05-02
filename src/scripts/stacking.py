import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import gc
import pandas as pd
from tqdm import tqdm
import click
from evaluate import *
from utils import *

@click.command()
@click.option('--ont', '-ont')

def main(ont):
  save_path = f'../models/stacking-{ont}.pt'

  y_val = pd.read_csv(f'../base/{ont}_val.csv').iloc[:, 2:].values
  y_test = pd.read_csv(f'../base/{ont}_test.csv').iloc[:, 2:].values
  ontologies_names = pd.read_csv(f'../base/{ont}_val.csv').columns[2:].values
  ic_ont = pd.read_csv(f'../base/{ont}_ic.csv')
  ic = ic_ont.set_index('terms')['IC'].to_dict()
  output_neurons = len(ontologies_names)

  if ont == 'bp':
    ontology = generate_ontology('../base/go.obo', specific_space=True, name_specific_space='biological_process')
    root = 'GO:0008150'
  elif ont == 'cc':
    ontology = generate_ontology('../base/go.obo', specific_space=True, name_specific_space='cellular_component')
    root = 'GO:0005575'
  else:
    ontology = generate_ontology('../base/go.obo', specific_space=True, name_specific_space='molecular_function')
    root = 'GO:0003674'

  preds_val, preds_test = [], []
  for layer in ['mlp-36', 'mlp-35', 'mlp-34', 'mlp-24', 'mlp-23', 'mlp-22', 'resnet50']:
    preds_val.append(np.load(f'../preds/{ont}-{layer}-val.npy'))
    preds_test.append(np.load(f'../preds/{ont}-{layer}-test.npy'))

  input_val = torch.tensor(np.hstack(preds_val))
  mask_val = torch.ones((input_val.shape[0], len(preds_val))).float()
  labels_val = torch.tensor(y_val).float()

  input_test = torch.tensor(np.hstack(preds_test))
  mask_test = torch.ones((input_test.shape[0], len(preds_test))).float()
  labels_test = torch.tensor(y_test).float()

  dataset_val_original = TensorDataset(input_val, mask_val, labels_val)
  train_size = int(0.8 * len(dataset_val_original))
  val_size = len(dataset_val_original) - train_size
  train_dataset, val_dataset = random_split(dataset_val_original, [train_size, val_size])
  train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
  full_val_dataloader = DataLoader(dataset_val_original, batch_size=32, shuffle=False)

  test_dataset = TensorDataset(input_test, mask_test, labels_test)
  test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

  model = EnsembleNN(num_outputs=output_neurons, num_models=len(preds_val))
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
  criterion = ProteinLoss(weight_tensor=ic_ont.IC.values)

  best_eval = float('-inf')
  patience_counter = 0
  patience = 5
  epochs = 50

  for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_data, batch_mask, batch_labels in tqdm(train_dataloader):
      batch_data = batch_data.to(device)
      batch_mask = batch_mask.to(device)
      batch_labels = batch_labels.to(device)
      optimizer.zero_grad()
      outputs = model(batch_data, batch_mask)
      outputs = torch.clamp(outputs, min=0.0, max=1.0)
      outputs = torch.logit(outputs, eps=1e-6)
      loss = criterion(outputs, batch_labels)
      total_loss += loss.item()
      loss.backward()
      optimizer.step()

    avg_loss = total_loss / len(train_dataloader)

    model.eval()
    val_loss = 0
    y_true_val = []
    y_pred_val = []
    with torch.no_grad():
      for batch_data, batch_mask, batch_labels in tqdm(val_dataloader):
        batch_data = batch_data.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = batch_labels.to(device)
        outputs = model(batch_data, batch_mask)
        outputs = torch.clamp(outputs, min=0.0, max=1.0)
        y_true_val.append(batch_labels.cpu().numpy())
        y_pred_val.append(outputs.clone().cpu().numpy())
        outputs = torch.logit(outputs, eps=1e-6)
        loss = criterion(outputs, batch_labels)
        val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss}, Validation Loss: {avg_val_loss}')

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
  model = EnsembleNN(num_outputs=output_neurons, num_models=len(preds_val))
  model.load_state_dict(torch.load(save_path, weights_only=True))
  model.to(device)
  model.eval()

  predict_set(full_val_dataloader, model, device, "val", ont)
  predict_set(test_dataloader, model, device, "test", ont)

  del model
  gc.collect()
  torch.cuda.empty_cache()

def predict_set(set_dataloader, model, device, set_name, ont):
  y_pred = []
  with torch.no_grad():
    for batch_data, batch_mask, batch_labels in tqdm(set_dataloader):
      batch_data = batch_data.to(device)
      batch_mask = batch_mask.to(device)
      batch_labels = batch_labels.to(device)
      outputs = model(batch_data, batch_mask)
      outputs = torch.clamp(outputs, min=0.0, max=1.0)
      y_pred.append(outputs.clone().cpu().numpy())

  f_pred = []
  for i in y_pred:
    for j in i:
      f_pred.append(j)
  f_pred = np.array(f_pred)
  np.save(f'../preds/{ont}-stacking-{set_name}.npy', f_pred)

if __name__ == '__main__':
  main()
