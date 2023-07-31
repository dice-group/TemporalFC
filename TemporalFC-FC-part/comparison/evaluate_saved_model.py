import pandas as pd

from data import Data
import torch
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn as nn
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from scipy import stats
import matplotlib.pyplot as plt
from utils.static_funcs import calculate_wilcoxen_score

class BaselineEmdeddingsOnlyModel(torch.nn.Module):
    def __init__(self, embedding_dim, num_entities, num_relations,dataset):
        super().__init__()
        self.ent_embeddings = dataset.emb_entities
        self.rel_embeddings = dataset.emb_relation
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        # self.loss = torch.nn.BCELoss()

        self.shallom_width = int(25.6 * self.embedding_dim)
        self.shallom_width2 = int(12.8 * self.embedding_dim)
        self.shallom_width3 = int(1 * self.embedding_dim)

        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)

        self.entity_embeddings.load_state_dict({'weight':torch.tensor(self.convrt_embeddings(num_entities,embedding_dim,self.ent_embeddings))})

        self.relation_embeddings.load_state_dict({'weight': torch.tensor(self.convrt_embeddings(num_relations,embedding_dim,self.rel_embeddings))})
        self.entity_embeddings.weight.requires_grad = False
        self.relation_embeddings.weight.requires_grad = False

        self.shallom = nn.Sequential(torch.nn.Linear(self.embedding_dim * 3, self.shallom_width3),
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
        # self.shallom = nn.Sequential(torch.nn.Linear(self.embedding_dim * 3, self.shallom_width),
        #                              nn.Dropout(0.50),
        #                              # torch.nn.Linear(self.shallom_width3, self.shallom_width2),
        #                              # nn.Dropout(0.50),
        #                              # torch.nn.Linear(self.shallom_width2, self.shallom_width),
        #                              # nn.Dropout(0.50),
        #                              # nn.BatchNorm1d(self.shallom_width),
        #                              # nn.Dropout(0.50),
        #                              nn.ReLU(self.shallom_width),
        #                              nn.Dropout(0.50),
        #                              torch.nn.Linear(self.shallom_width, 1))

    def convrt_embeddings(self,num_entities,embedding_dim,embeddings):
        weights_matrix = np.zeros((num_entities, embedding_dim))
        words_found = 0
        for i, word in enumerate(embeddings):
            try:
                weights_matrix[i] = word
                words_found += 1
            except KeyError:
                print('test')
                exit(1)
        return weights_matrix

    def forward(self, e1_idx, rel_idx, e2_idx,sen_idx="", type="training"):
        emb_head_real = self.entity_embeddings(e1_idx)
        emb_rel_real = self.relation_embeddings(rel_idx)
        emb_tail_real = self.entity_embeddings(e2_idx)
        x = torch.cat([emb_head_real, emb_rel_real, emb_tail_real], 1)
        x2 = self.shallom(x)
        x3 = torch.sigmoid(x2)
        return x3


class HybridModel(torch.nn.Module):
    def __init__(self, embedding_dim, num_entities, num_relations, dataset):
        super().__init__()
        self.ent_embeddings = dataset.emb_entities
        self.rel_embeddings = dataset.emb_relation

        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.loss = torch.nn.BCELoss()
        self.sentence_dim=768*3

        self.shallom_width = int(25.6 * self.embedding_dim)
        self.shallom_width2 = int(12.8 * self.embedding_dim)
        self.shallom_width3 = int(1 * self.embedding_dim)
        self.sen_embeddings_train =  dataset.emb_sentences_train
        self.sen_embeddings_test = dataset.emb_sentences_test
        self.sen_embeddings_valid = dataset.emb_sentences_valid

        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.sentence_embeddings_train = nn.Embedding(len(self.sen_embeddings_train), self.sentence_dim)
        self.sentence_embeddings_test = nn.Embedding(len(self.sen_embeddings_test), self.sentence_dim)
        self.sentence_embeddings_valid = nn.Embedding(len(self.sen_embeddings_valid), self.sentence_dim)

        self.entity_embeddings.load_state_dict(
            {'weight': torch.tensor(self.convrt_embeddings(num_entities, embedding_dim, self.ent_embeddings))})

        self.relation_embeddings.load_state_dict(
            {'weight': torch.tensor(self.convrt_embeddings(num_relations, embedding_dim, self.rel_embeddings))})

        self.sentence_embeddings_train.load_state_dict(
            {'weight': torch.tensor(self.convrt_embeddings(len(self.sen_embeddings_train), self.sentence_dim, self.sen_embeddings_train))})

        self.sentence_embeddings_test.load_state_dict(
            {'weight': torch.tensor(
                self.convrt_embeddings(len(self.sen_embeddings_test), self.sentence_dim, self.sen_embeddings_test))})
        self.sentence_embeddings_valid.load_state_dict(
            {'weight': torch.tensor(
                self.convrt_embeddings(len(self.sen_embeddings_valid), self.sentence_dim, self.sen_embeddings_valid))})
        self.entity_embeddings.weight.requires_grad = False
        self.relation_embeddings.weight.requires_grad = False
        self.sentence_embeddings_test.weight.requires_grad = False
        self.sentence_embeddings_valid.weight.requires_grad = False
        self.sentence_embeddings_train.weight.requires_grad = False

        self.shallom = nn.Sequential(torch.nn.Linear(self.embedding_dim * 3, self.shallom_width),
                                     nn.BatchNorm1d(self.shallom_width),
                                     nn.ReLU(),
                                     nn.Dropout(0.50),
                                     torch.nn.Linear(self.shallom_width, self.shallom_width))

        self.shallom_sentence = nn.Sequential(torch.nn.Linear(self.sentence_dim , self.shallom_width),
                                     nn.BatchNorm1d(self.shallom_width),
                                     nn.ReLU(),
                                    nn.Dropout(0.50),
                                     torch.nn.Linear(self.shallom_width, self.shallom_width))

        self.classification = nn.Sequential(torch.nn.Linear(self.shallom_width * 2, self.shallom_width),
                                     nn.BatchNorm1d(self.shallom_width),
                                     nn.ReLU(),
                                     nn.Dropout(0.50),
                                     torch.nn.Linear(self.shallom_width, 1))

    def convrt_embeddings(self,num_entities,embedding_dim,embeddings):
        weights_matrix = np.zeros((num_entities, embedding_dim))
        words_found = 0
        for i, word in enumerate(embeddings):
            try:
                weights_matrix[i] = word
                words_found += 1
            except KeyError:
                print('test')
                exit(1)
        return weights_matrix

    def forward(self, e1_idx, rel_idx, e2_idx,sen_idx, type="training"):
        # print(sen_idx)
        emb_head_real = self.entity_embeddings(e1_idx)
        emb_rel_real = self.relation_embeddings(rel_idx)
        emb_tail_real = self.entity_embeddings(e2_idx)
        x = torch.cat([emb_head_real, emb_rel_real, emb_tail_real], 1)
        triplet_embedding = self.shallom(x)
        emb_sen =[]
        if type.__contains__("training"):
            emb_sen = self.sentence_embeddings_train(sen_idx)
        elif type.__contains__("valid"):
            emb_sen = self.sentence_embeddings_valid(sen_idx)
        else:
            emb_sen = self.sentence_embeddings_test(sen_idx)
        sentence_embedding = self.shallom_sentence(emb_sen)
        z = torch.cat([triplet_embedding,sentence_embedding],1)
        return torch.sigmoid(self.classification(z))




class HybridModelSetting2(torch.nn.Module):
    def __init__(self, embedding_dim, num_entities, num_relations, dataset):
        super().__init__()
        self.ent_embeddings = dataset.emb_entities
        self.rel_embeddings = dataset.emb_relation

        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        # self.loss = torch.nn.BCELoss()
        self.sentence_dim=768*3
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.shallom_width = int(25.6 * self.embedding_dim)
        self.sen_embeddings_train =  dataset.emb_sentences_train
        self.sen_embeddings_test = dataset.emb_sentences_test
        self.sen_embeddings_valid = dataset.emb_sentences_valid

        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.sentence_embeddings_train = nn.Embedding(len(self.sen_embeddings_train), self.sentence_dim)
        self.sentence_embeddings_test = nn.Embedding(len(self.sen_embeddings_test), self.sentence_dim)
        self.sentence_embeddings_valid = nn.Embedding(len(self.sen_embeddings_valid), self.sentence_dim)

        self.entity_embeddings.load_state_dict(
            {'weight': torch.tensor(self.convrt_embeddings(num_entities, embedding_dim, self.ent_embeddings))})

        self.relation_embeddings.load_state_dict(
            {'weight': torch.tensor(self.convrt_embeddings(num_relations, embedding_dim, self.rel_embeddings))})

        self.sentence_embeddings_train.load_state_dict(
            {'weight': torch.tensor(self.convrt_embeddings(len(self.sen_embeddings_train), self.sentence_dim, self.sen_embeddings_train))})

        self.sentence_embeddings_test.load_state_dict(
            {'weight': torch.tensor(
                self.convrt_embeddings(len(self.sen_embeddings_test), self.sentence_dim, self.sen_embeddings_test))})

        self.sentence_embeddings_valid.load_state_dict(
            {'weight': torch.tensor(
                self.convrt_embeddings(len(self.sen_embeddings_valid), self.sentence_dim, self.sen_embeddings_valid))})

        self.classification = nn.Sequential(
            torch.nn.Linear(self.embedding_dim * 3 + self.sentence_dim, self.shallom_width),
            nn.BatchNorm1d(self.shallom_width),
            nn.Dropout(.20),
            nn.ReLU(),
            nn.Dropout(.20),
            torch.nn.Linear(self.shallom_width, 1))

    def convrt_embeddings(self,num_entities,embedding_dim,embeddings):
        weights_matrix = np.zeros((num_entities, embedding_dim))
        words_found = 0
        for i, word in enumerate(embeddings):
            try:
                weights_matrix[i] = word
                words_found += 1
            except KeyError:
                print('test')
                exit(1)
        return weights_matrix

    def forward(self, e1_idx, rel_idx, e2_idx,sen_idx, type="training"):
        # print(sen_idx)
        emb_head_real = self.entity_embeddings(e1_idx)
        emb_rel_real = self.relation_embeddings(rel_idx)
        emb_tail_real = self.entity_embeddings(e2_idx)
        x = torch.cat([emb_head_real, emb_rel_real, emb_tail_real], 1)
        # triplet_embedding = self.shallom(x)
        emb_sen =[]
        if type.__contains__("training"):
            emb_sen = self.sentence_embeddings_train(sen_idx)
        elif type.__contains__("valid"):
            emb_sen = self.sentence_embeddings_valid(sen_idx)
        else:
            emb_sen = self.sentence_embeddings_test(sen_idx)
        # sentence_embedding = self.shallom_sentence(emb_sen)
        z = torch.cat([emb_head_real, emb_rel_real, emb_tail_real,emb_sen],1)
        z = self.dropout(z)
        return torch.sigmoid(self.classification(z))




bpdp_ds = False
epochs = 200
datasets_class = ["property/","domain/","domainrange/","mix/","random/","range/"]
cls2 = datasets_class[0]
emb_model = "TransE"

if bpdp_ds:
    cls2 = "bpdp"
complete_data_set = ["complete_dataset"]
properties_split = ["deathPlace/","birthPlace/","author/","award/","foundationPlace/","spouse/","starring/","subsidiary/"]
# make it true or false
prop_split = False
complete_data = False
clss = datasets_class

if complete_data:
    clss = complete_data_set
elif prop_split:
    clss = properties_split

methods = ["emb-only","hybrid"]
df_test = pd.DataFrame()
df_train = pd.DataFrame()


for cls in methods:
# else:
    print("processing:" + cls)
    method = cls #emb-only  #hybrid
    path_dataset_folder = '../dataset/'
    if bpdp_ds == True:
        path_dataset_folder = '../dataset/data/bpdp/'
        dataset = Data(data_dir=path_dataset_folder, subpath=None, complete_data=complete_data, emb_typ=emb_model,
                       emb_file="../", bpdp_dataset=True)
    else:
        dataset = Data(data_dir=path_dataset_folder, subpath=cls2, emb_typ= emb_model,emb_file = "../")
    # dataset1 = Data(data_dir=path_dataset_folder, subpath= cls, prop=cls)
    num_entities, num_relations = len(dataset.entities), len(dataset.relations)

    if method == "emb-only":
        model = BaselineEmdeddingsOnlyModel(embedding_dim=100, num_entities=num_entities, num_relations=num_relations,
                                            dataset=dataset)
    else:
        model = HybridModel(embedding_dim=100, num_entities=num_entities, num_relations=num_relations,
                                    dataset=dataset)
    # path_dataset_folder = '../dataset/'
    model.load_state_dict(torch.load(path_dataset_folder+'models/'+emb_model+'-'+cls2.replace("/","-")+'-' +method+'-' + str(epochs) + '.pth'))
    model.eval()
    print(model)


    # Train F1 train dataset
    X_train = np.array(dataset.idx_train_data)[:, :5]
    y_train = np.array(dataset.idx_train_data)[:, -2]
    X_train_tensor = torch.Tensor(X_train).long()

    jj = np.arange(0, len(X_train))
    x_data = torch.tensor(jj)

        # X_sen_train_tensor = torch.Tensor(X_sen_train).long()
    idx_s, idx_p, idx_o = X_train_tensor[:, 0], X_train_tensor[:, 1], X_train_tensor[:, 2]
    prob = model(idx_s, idx_p, idx_o,x_data).flatten()
    pred = (prob > 0.50).float()
    pred = pred.data.detach().numpy()
    print('Acc score on train data', accuracy_score(y_train, pred))
    print('report:', classification_report(y_train,pred))
    df_train[cls] = pred



    # Train F1 test dataset
    X_test = np.array(dataset.idx_test_data)[:, :5]
    y_test = np.array(dataset.idx_test_data)[:, -2]
    X_test_tensor = torch.Tensor(X_test).long()
    idx_s, idx_p, idx_o = X_test_tensor[:, 0], X_test_tensor[:, 1], X_test_tensor[:, 2]

    jj = np.arange(0, len(X_test))
    x_data = torch.tensor(jj)

    prob = model(idx_s, idx_p, idx_o,x_data,"testing").flatten()
    pred = (prob > 0.50).float()
    pred = pred.data.detach().numpy()
    print('Acc score on test data', accuracy_score(y_test, pred))
    print('report:', classification_report(y_test,pred))
    df_test[cls] = pred
    exit(1)

# stats test
calculate_wilcoxen_score(df_train,"train")
calculate_wilcoxen_score(df_test,"test")

