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

class TemporalPredictionModel(BaseKGE):
    def __init__(self, args): #embedding_dim, num_entities, num_relations,
        super().__init__(args)
        self.name = 'TemporalModel'
        self.ent_embeddings = args.dataset.emb_entities
        self.rel_embeddings = args.dataset.emb_relation
        self.tim_embeddings = args.dataset.emb_time

        self.embedding_dim = args.embedding_dim
        self.num_entities = args.num_entities
        self.num_relations = args.num_relations
        self.num_times = args.num_times
        # self.loss = torch.nn.CosineSimilarity()
        self.loss = torch.nn.CrossEntropyLoss()

        for i, word in enumerate(self.ent_embeddings):
            self.embedding_dim = len(word)
            break
        for i, word in enumerate(self.rel_embeddings):
            self.embedding_dim_rel = len(word)
            break

        for i, word in enumerate(self.tim_embeddings):
            self.embedding_dim_tim = len(word)
            break



        self.shallom_width = int(25.6 * self.embedding_dim)
        self.shallom_width2 = int(12.8 * self.embedding_dim)
        self.shallom_width3 = int(1 * self.embedding_dim)

        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim_rel)
        self.time_embeddings = nn.Embedding(self.num_times, self.embedding_dim_tim)

        if args.include_veracity == True:
            self.veracity_score_train1 = args.dataset.veracity_train
            self.veracity_score_test1 = args.dataset.veracity_test
            self.veracity_score_valid1 = args.dataset.veracity_valid

            self.veracity_score_train = nn.Embedding(len(self.veracity_score_train1), 1)
            self.veracity_score_test = nn.Embedding(len(self.veracity_score_test1), 1)
            self.veracity_score_valid = nn.Embedding(len(self.veracity_score_valid1), 1)



        self.entity_embeddings.load_state_dict({'weight':torch.tensor(self.convrt_embeddings(args.num_entities,self.embedding_dim,self.ent_embeddings))})

        self.relation_embeddings.load_state_dict({'weight': torch.tensor(self.convrt_embeddings(args.num_relations,self.embedding_dim_rel,self.rel_embeddings))})

        self.time_embeddings.load_state_dict({'weight': torch.tensor(self.convrt_embeddings(args.num_times,self.embedding_dim_tim,self.tim_embeddings))})

        self.entity_embeddings.weight.requires_grad = False
        self.relation_embeddings.weight.requires_grad = False
        self.time_embeddings.weight.requires_grad = False

        if args.include_veracity == True:
            self.veracity_score_train.weight.requires_grad = False
            self.veracity_score_test.weight.requires_grad = False
            self.veracity_score_valid.weight.requires_grad = False

        # + self.embedding_dim_tim
        self.shallom = nn.Sequential(torch.nn.Linear(self.embedding_dim * 2 + self.embedding_dim_rel  , self.shallom_width),
                                     nn.Dropout(0.20),
                                     nn.BatchNorm1d(self.shallom_width),
                                     # nn.Dropout(0.50),
                                     # nn.ReLU(self.shallom_width3),
                                     # torch.nn.Linear(self.shallom_width, self.shallom_width2),
                                     nn.Dropout(0.20),
                                     torch.nn.Linear(self.shallom_width, self.num_times))

    # self.embedding_dim_tim

    def forward_triples(self, e1_idx, rel_idx, e2_idx,t_idx, v_idx="", type="training"):
        emb_head_real = self.entity_embeddings(e1_idx)
        emb_rel_real = self.relation_embeddings(rel_idx)
        emb_tail_real = self.entity_embeddings(e2_idx)
        # emb_tim_real = self.time_embeddings(t_idx)
        # veracity exculded because not needed in time prediction anymore
        # ver_score = 0.0
        # if type.__contains__("training"):
        #     ver_score = self.veracity_score_train(v_idx)
        # elif type.__contains__("valid"):
        #     ver_score = self.veracity_score_valid(v_idx)
        # else:
        #     ver_score = self.veracity_score_test(v_idx)
        x = torch.cat([emb_head_real, emb_rel_real, emb_tail_real], 1)
        x2 = self.shallom(x)
        # x3 = torch.softmax(x2, dim=0)
        return x2
