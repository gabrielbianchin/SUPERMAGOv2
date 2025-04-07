import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision.transforms import Resize, Compose, Normalize, ToTensor
import pandas as pd
from tqdm import tqdm
import gc
from utils import *
from evaluate import *
import click

@click.command()
@click.option('--ont', '-ont')

def main(ont):
  save_path = f'../models/resnet50-{ont}.pt'

  train = pd.read_csv(f'../base/{ont}_train.csv')
  val = pd.read_csv(f'../base/{ont}_val.csv')
  test = pd.read_csv(f'../base/{ont}_test.csv')
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

  img_size = (64, 56)

  y_train = train.iloc[:, 2:].values
  y_val = val.iloc[:, 2:].values
  y_test = test.iloc[:, 2:].values

  _transforms = Compose([
    Resize(img_size),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  train_dataset = ImageDataset(train.iloc[:, 0].values, y_train, transform=_transforms)
  val_dataset = ImageDataset(val.iloc[:, 0].values, y_val, transform=_transforms)
  test_dataset = ImageDataset(test.iloc[:, 0].values, y_test, transform=_transforms)

  train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
  test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = ResNet50(num_outputs=output_neurons).to(device)

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
      pixel_values = batch["pixel_values"].to(device)
      labels = batch["label"].to(device)
      outputs = model(pixel_values)
      loss = criterion(outputs, labels)
      total_loss += loss.item()
      loss.backward()
      optimizer.step()

    avg_loss = total_loss / len(train_dataloader)

    model.eval()
    val_loss = 0
    y_true_val = []
    y_pred_val = []
    with torch.no_grad():
      for val_batch in tqdm(val_dataloader):
        pixel_values = val_batch["pixel_values"].to(device)
        labels = val_batch["label"].to(device)
        outputs = model(pixel_values)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        preds_val = torch.sigmoid(outputs)
        y_true_val.append(val_batch["label"].cpu().numpy())
        y_pred_val.append(preds_val.cpu().numpy())

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
  model = ResNet50(num_outputs=output_neurons)
  model.load_state_dict(torch.load(save_path, weights_only=True))
  model.to(device)
  model.eval()

  predict_set(val_dataloader, model, device, "val", ont)
  predict_set(test_dataloader, model, device, "test", ont)

  del model
  gc.collect()
  torch.cuda.empty_cache()

def predict_set(set_dataloader, model, device, set_name, ont):
  y_pred = []
  with torch.no_grad():
    for batch in tqdm(set_dataloader):
      pixel_values = batch["pixel_values"].to(device)
      labels = batch["label"].to(device)
      outputs = model(pixel_values)
      pred = torch.sigmoid(outputs)
      y_pred.append(pred.cpu().numpy())

  f_pred = []
  for i in y_pred:
    for j in i:
      f_pred.append(j)
  f_pred = np.array(f_pred)
  np.save(f'../preds/{ont}-resnet50-{set_name}.npy', f_pred)

if __name__ == '__main__':
  main()
