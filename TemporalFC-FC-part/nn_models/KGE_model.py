from .base_model import *

import torch.nn as nn
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

class KGEModel(BaseKGE):
    def __init__(self, args): #embedding_dim, num_entities, num_relations,
        super().__init__()
        self.name = 'KGEModel'
        self.ent_embeddings = args.dataset.emb_entities
        self.rel_embeddings = args.dataset.emb_relation
        self.embedding_dim = args.embedding_dim
        self.num_entities = args.num_entities
        self.num_relations = args.num_relations
        self.loss = torch.nn.BCELoss()

        for i, word in enumerate(self.ent_embeddings):
            self.embedding_dim = len(word)
            break
        for i, word in enumerate(self.rel_embeddings):
            self.embedding_dim_rel = len(word)
            break

        self.shallom_width = int(25.6 * self.embedding_dim)
        self.shallom_width2 = int(12.8 * self.embedding_dim)
        self.shallom_width3 = int(1 * self.embedding_dim)

        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim_rel)

        self.entity_embeddings.load_state_dict({'weight':torch.tensor(self.convrt_embeddings(args.num_entities,self.embedding_dim,self.ent_embeddings))})

        self.relation_embeddings.load_state_dict({'weight': torch.tensor(self.convrt_embeddings(args.num_relations,self.embedding_dim_rel,self.rel_embeddings))})
        self.entity_embeddings.weight.requires_grad = False
        self.relation_embeddings.weight.requires_grad = False


        self.shallom = nn.Sequential(torch.nn.Linear(self.embedding_dim * 2 + self.embedding_dim_rel, self.shallom_width3),
                                     nn.Dropout(0.50),
                                     # torch.nn.Linear(self.shallom_width3, self.shallom_width2),
                                     # nn.Dropout(0.50),
                                     # torch.nn.Linear(self.shallom_width2, self.shallom_width),
                                     # nn.Dropout(0.50),
                                     nn.BatchNorm1d(self.shallom_width3),
                                     nn.Dropout(0.50),
                                     nn.ReLU(self.shallom_width3),
                                     nn.Dropout(0.50),
                                     torch.nn.Linear(self.shallom_width3, 1))


    def forward_triples(self, e1_idx, rel_idx, e2_idx,sen_idx="", type="training"):
        emb_head_real = self.entity_embeddings(e1_idx)
        emb_rel_real = self.relation_embeddings(rel_idx)
        emb_tail_real = self.entity_embeddings(e2_idx)
        x = torch.cat([emb_head_real, emb_rel_real, emb_tail_real], 1)
        x2 = self.shallom(x)
        x3 = torch.sigmoid(x2)
        return x3
