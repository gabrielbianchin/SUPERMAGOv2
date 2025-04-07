import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import models

class ImageDataset(Dataset):
  def __init__(self, names, labels, transform=None):
    self.names = names
    self.labels = labels
    self.transform = transform

  def __len__(self):
    return len(self.names)

  def generate_img(self, this_id):
    e1 = np.load(f'../embs/img-{this_id}-36.npy')
    e2 = np.load(f'../embs/img-{this_id}-35.npy')
    e3 = np.load(f'../embs/img-{this_id}-34.npy')
    t1 = np.load(f'../embs/img-{this_id}-24.npy')
    t2 = np.load(f'../embs/img-{this_id}-23.npy')
    t3 = np.load(f'../embs/img-{this_id}-22.npy')
    c1 = np.concatenate([e1, t1]).reshape(64, 56)
    c2 = np.concatenate([e2, t2]).reshape(64, 56)
    c3 = np.concatenate([e3, t3]).reshape(64, 56)
    X = np.transpose(np.array([c1, c2, c3]).astype(np.uint8), (1, 2, 0))
    img = Image.fromarray(X)
    return img

  def __getitem__(self, idx):
    name = self.names[idx]
    image = self.generate_img(name)
    if self.transform:
      image = self.transform(image)
    label = torch.tensor(self.labels[idx], dtype=torch.float)
    return {"pixel_values": image, "label": label}

class MLPDataset(Dataset):
  def __init__(self, names_queries, names_subject, bitscores, labels, layer):
    self.names_queries = names_queries
    self.names_subject = names_subject
    self.bitscores = bitscores
    self.labels = labels
    self.layer = layer

  def __len__(self):
    return len(self.names_queries)

  def __getitem__(self, idx):
    name_query = self.names_queries[idx]
    name_subject = self.names_subject[idx]
    label = self.labels[idx]
    bitscores = self.bitscores[idx]
    w = torch.tensor(np.array(bitscores)/sum(bitscores)).view(-1, 1)
    layer = self.layer

    features = []
    features.append(torch.tensor(np.load(f'../embs/{name_query}-{layer}.npy'), dtype=torch.float))
    if len(name_subject) == 0:
      features.append(torch.tensor(np.load(f'../embs/{name_query}-{layer}.npy'), dtype=torch.float))
    else:
      aux = []
      for n in name_subject:
        aux.append(torch.tensor(np.load(f'../embs/{n}-{layer}.npy'), dtype=torch.float))
      aux = torch.sum(torch.vstack(aux) * w, axis=0)
      features.append(aux.clone().detach().float())

    return {"features": features, "label": label}

class ProteinLoss(nn.Module):
  def __init__(
    self,
    weight_tensor,
    device='cuda'
  ):
    super(ProteinLoss, self).__init__()

    self.device = device
    self.weight_tensor = torch.from_numpy(weight_tensor).float().to(self.device)

  def forward(self, y_pred, y_true):
    sig_y_pred = torch.sigmoid(y_pred)
    crossentropy_loss = self.multilabel_categorical_crossentropy(y_pred, y_true)
    go_term_centric_loss = self.weight_f1_loss(sig_y_pred, y_true, centric='go')
    protein_centric_loss = self.weight_f1_loss(sig_y_pred, y_true, centric='protein')
    total_loss = crossentropy_loss * protein_centric_loss * go_term_centric_loss
    return total_loss

  def multilabel_categorical_crossentropy(self, y_pred, y_true):

    # Modify predicted probabilities based on true labels
    y_pred = (1 - 2 * y_true) * y_pred

    # Adjust predicted probabilities
    y_pred_neg = y_pred - y_true * 1e16
    y_pred_pos = y_pred - (1 - y_true) * 1e16

    # Concatenate zeros tensor
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)

    # Compute logsumexp along the class dimension (dim=1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    total_loss = neg_loss + pos_loss

    return torch.mean(total_loss)

  def weight_f1_loss(self, y_pred, y_true, beta=1.0, centric='protein'):
    weight_tensor = self.weight_tensor.to(self.device)

    dim = 1 if centric == 'protein' else 0

    tp = torch.sum(y_true * y_pred * weight_tensor, dim=dim).to(y_pred.device)
    fp = torch.sum((1 - y_true) * y_pred * weight_tensor, dim=dim).to(y_pred.device)
    fn = torch.sum(y_true * (1 - y_pred) * weight_tensor, dim=dim).to(y_pred.device)

    precision = tp / (tp + fp + 1e-16)
    recall = tp / (tp + fn + 1e-16)

    mean_precision = torch.mean(precision)
    mean_recall = torch.mean(recall)
    f1 = self.f1_score(mean_precision, mean_recall, beta=beta)

    return 1 - f1

  def f1_score(self, precision, recall, beta=0.5, eps=1e-16):
    f1 = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + eps)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    return f1

class ResNet50(nn.Module):
  def __init__(self, num_outputs):
    super(ResNet50, self).__init__()
    self.base = models.resnet50(pretrained=True)
    self.base.fc = nn.Linear(in_features=self.base.fc.in_features, out_features=num_outputs)

  def forward(self, x):
    return self.base(x)

class MLP(nn.Module):
  def __init__(self, num_outputs, shapes):
    super(MLP, self).__init__()

    self.branches = nn.ModuleList([self.build_branch(shape) for shape in shapes])

    self.classifier = nn.Sequential(
      nn.Linear(len(shapes) * 1024, 2048),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(2048, num_outputs)
    )

  def build_branch(self, input_size):
    return nn.Sequential(
      nn.BatchNorm1d(input_size),
      nn.Linear(input_size, 1024),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.BatchNorm1d(1024)
    )

  def forward(self, inputs):
    processed_inputs = []
    for i, inp in enumerate(inputs):
      emb = self.branches[i](inp)
      processed_inputs.append(emb)

    concatenated = torch.cat(processed_inputs, dim=1)
    x = self.classifier(concatenated)
    return x

class NormalizedWeightedSumWithMask(nn.Module):
  def __init__(self, input_dim):
    super(NormalizedWeightedSumWithMask, self).__init__()
    self.custom_weights = nn.Parameter(torch.rand(input_dim))

  def forward(self, inputs, mask):
    masked_weights = torch.where(mask > 0, self.custom_weights, torch.full_like(self.custom_weights, -1e9))
    normalized_weights = F.softmax(masked_weights, dim=1)
    weighted_sum = torch.sum(inputs * normalized_weights, dim=1)
    return weighted_sum

class EnsembleNN(nn.Module):
  def __init__(self, num_outputs, num_models):
    super(EnsembleNN, self).__init__()
    self.num_outputs = num_outputs
    self.num_models = num_models
    self.weighted_sums = nn.ModuleList([NormalizedWeightedSumWithMask(num_models) for _ in range(num_outputs)])

  def forward(self, x, mask):
    outputs = []
    for i in range(self.num_outputs):
      preds = [x[:, j * self.num_outputs + i] for j in range(self.num_models)]
      output = self.weighted_sums[i](torch.stack(preds).T, mask)
      outputs.append(output)

    return torch.stack(outputs).T
