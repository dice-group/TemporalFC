import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional import accuracy
from typing import List, Any, Tuple
from torch.nn.init import xavier_normal_
from torch import Tensor as tf
import torch.nn as nn
import pandas as pd

from numpy.random import RandomState
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import random
import numpy as np
# from pytorchtools import EarlyStopping
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

from sklearn.metrics import make_scorer


from scipy.spatial.distance import cosine


class BaseKGE(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.name = 'Not init'

        # added by umair to make time embeddings as a label...todo-> find a solution to move it in main class
        self.embedding_dim = args.embedding_dim
        if str(args.model).startswith("temporal"):
            self.tim_embeddings = args.dataset.emb_time
            self.num_times = args.num_times
            for i, word in enumerate(self.tim_embeddings):
                self.embedding_dim_tim = len(word)
                break
            self.time_embeddings = nn.Embedding(self.num_times, self.embedding_dim_tim)
            self.time_embeddings.load_state_dict({'weight': torch.tensor(self.convrt_embeddings(args.num_times, self.embedding_dim_tim, self.tim_embeddings))})


    def convrt_embeddings(self,num_entities,embedding_dim,embeddings):
        weights_matrix = np.zeros((num_entities, embedding_dim))
        words_found = 0
        if isinstance(embeddings, pd.DataFrame):
            ii = 0
            for i, word in embeddings.iterrows():
                try:
                    if len(word) == embedding_dim:
                        weights_matrix[ii] = word
                    else:
                        weights_matrix[ii] = word.detach().numpy()
                    ii = ii +1
                    words_found += 1
                except KeyError:
                    print('test')
                    exit(1)
            return weights_matrix

        else:
            for i, word in enumerate(embeddings):
                try:
                    if len(word) == 2305:
                        weights_matrix[i] = word[:-1]
                    else:
                        weights_matrix[i] = word.detach().numpy()
                    words_found += 1
                except KeyError:
                    print('test')
                    exit(1)
            return weights_matrix

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def loss_function(self, y_hat, y):
        return self.loss(y_hat, y)

    def forward_triples(self, *args, **kwargs):
        raise ValueError(f'MODEL:{self.name} does not have forward_triples function')

    # self, e1_idx, rel_idx, e2_idx, sen_idx, type = "training"):
    def forward(self, x):
        if len(x) == 5:
            e1_idx, rel_idx, e2_idx, t_idx, s_idx, v_idx = x[0], x[1], x[2], x[3], x[4], x[5]
            return self.forward_triples(e1_idx, rel_idx, e2_idx,t_idx, s_idx, v_idx)
        else:
            raise ValueError('Not valid input')

    def training_step(self, batch, batch_idx):
        idx_s, idx_p, idx_o, t_data, s_data, v_data, label = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]
        # Label conversion
        # for time prediction only the following unupre0d
        # label = t_data
        # label = self.time_embeddings(t_data)
        # 2. Forward pass
        pred = self.forward_triples(idx_s, idx_p, idx_o, t_data, s_data, v_data, "training")
        # 3. Compute Loss
        loss = self.loss_function(pred.flatten(), torch.tensor(label,dtype=torch.float))
        # applying L2 regularization here
        # Replaces pow(2.0) with abs() for L1 regularization
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum()
                      for p in self.parameters())

        loss = loss + l2_lambda * l2_norm
        # preds2 = self.get_min_cosine_similarity(label, pred)
        # targets = self.get_target_indices(label, minn)
        # train_accuracy = accuracy(torch.LongTensor(preds2), t_data)
        train_accuracy = accuracy(pred, label)
        return {'acc': train_accuracy, 'loss': loss}

    def training_epoch_end(self, outputs) -> None:
        x = [[x['acc'], x['loss']] for x in outputs]
        avg_train_acc, avg_train_loss = torch.tensor(x).mean(dim=0)[:]
        self.log('avg_train_loss_per_epoch', avg_train_loss, on_epoch=True, prog_bar=True)
        self.log('avg_train_acc_per_epoch', avg_train_acc, on_epoch=True, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        idx_s, idx_p, idx_o, t_data,s_data, v_data, label = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]
        # Label conversion
        # label = t_data
        # label = self.time_embeddings(t_data)
        # 2. Forward pass
        pred = self.forward_triples(idx_s, idx_p, idx_o, t_data, s_data, v_data, "valid")


        # Find the Loss
        loss = self.loss_function(pred.flatten(), torch.tensor(label,dtype=torch.float))
        # applying L2 regularization here
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum()
                      for p in self.parameters())

        loss = loss + l2_lambda * l2_norm
        # preds2 = self.get_min_cosine_similarity2(label.numpy(), pred.numpy())
        # targets = self.get_target_indices(label, minn)
        val_accuracy = accuracy(pred, label)

        return {'val_acc': val_accuracy, 'val_loss': loss}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        x = [[x['val_acc'], x['val_loss']] for x in outputs]
        avg_val_acc, avg_val_loss = torch.tensor(x).mean(dim=0)[:]
        self.log('avg_val_loss_per_epoch', avg_val_loss, on_epoch=True, prog_bar=True)
        self.log('avg_val_acc_per_epoch', avg_val_acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        idx_s, idx_p, idx_o, t_data, s_data, v_data, label = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]
        # Label conversion
        # label = t_data
        # label = self.time_embeddings(t_data)
        # 2. Forward pass
        pred = self.forward_triples(idx_s, idx_p, idx_o, t_data, s_data, v_data, "test")

        # test_accuracy = accuracy(pred, label)
        # preds2 = self.get_min_cosine_similarity(label, pred)
        # targets = self.get_target_indices(label, minn)
        test_accuracy = accuracy(pred, label)

        return {'test_accuracy': test_accuracy}

    def test_epoch_end(self, outputs: List[Any]):
        avg_test_accuracy = torch.stack([x['test_accuracy'] for x in outputs]).mean()
        self.log('avg_test_accuracy', avg_test_accuracy, on_epoch=True, prog_bar=True)
