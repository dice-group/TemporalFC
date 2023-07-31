from data import Data
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
        # self.dropout = nn.Dropout(0.20)
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
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

        self.classification = nn.Sequential(torch.nn.Linear(self.embedding_dim * 3+self.sentence_dim , self.shallom_width),
                                     nn.Dropout(.50),
                                     torch.nn.Linear(self.shallom_width3, self.shallom_width2),
                                     nn.Dropout(0.50),
                                     torch.nn.Linear(self.shallom_width2, self.shallom_width),
                                     nn.Dropout(0.50),
                                     nn.BatchNorm1d(self.shallom_width),
                                     nn.ReLU(),
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
        # z = self.dropout(z)
        return torch.sigmoid(self.classification(z))

class BaselineEmdeddingsOnlyModel(torch.nn.Module):
    def __init__(self, embedding_dim, num_entities, num_relations,dataset):
        super().__init__()
        self.ent_embeddings = dataset.emb_entities
        self.rel_embeddings = dataset.emb_relation
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations

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

        self.entity_embeddings.load_state_dict({'weight':torch.tensor(self.convrt_embeddings(num_entities,embedding_dim,self.ent_embeddings))})

        self.relation_embeddings.load_state_dict({'weight': torch.tensor(self.convrt_embeddings(num_relations,self.embedding_dim_rel,self.rel_embeddings))})
        self.entity_embeddings.weight.requires_grad = False
        self.relation_embeddings.weight.requires_grad = False


        self.shallom = nn.Sequential(torch.nn.Linear((self.embedding_dim * 2)+self.embedding_dim_rel, self.shallom_width3),
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

class AnaBaselineEmdeddingsOnlyModel(torch.nn.Module):
    def __init__(self, embedding_dim, num_entities, num_relations,dataset):
        super().__init__()
        self.name = 'Hybrid'
        self.ent_embeddings = dataset.emb_entities
        self.rel_embeddings = dataset.emb_relation

        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.loss = torch.nn.BCELoss()
        self.sentence_dim =768*3

        for i, word in enumerate(self.ent_embeddings):
            self.embedding_dim = len(word)
            break
        for i, word in enumerate(self.rel_embeddings):
            self.embedding_dim_rel = len(word)
            break

        self.shallom_width = int(25.6 * self.embedding_dim)
        self.shallom_width2 = int(12.8 * self.embedding_dim)
        self.shallom_width3 = int(1 * self.embedding_dim)
        self.sen_embeddings_train = dataset.emb_sentences_train
        self.sen_embeddings_test = dataset.emb_sentences_test
        self.sen_embeddings_valid = dataset.emb_sentences_valid
        self.copaal_veracity_score_train1 = dataset.copaal_veracity_train
        self.copaal_veracity_score_test1 = dataset.copaal_veracity_test
        self.copaal_veracity_score_valid1 = dataset.copaal_veracity_valid

        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim_rel)
        self.sentence_embeddings_train = nn.Embedding(len(self.sen_embeddings_train), self.sentence_dim)
        self.sentence_embeddings_test = nn.Embedding(len(self.sen_embeddings_test), self.sentence_dim)
        self.sentence_embeddings_valid = nn.Embedding(len(self.sen_embeddings_valid), self.sentence_dim)
        self.copaal_veracity_score_train = nn.Embedding(len(self.copaal_veracity_score_train1), 1)
        self.copaal_veracity_score_test = nn.Embedding(len(self.copaal_veracity_score_test1), 1)
        self.copaal_veracity_score_valid = nn.Embedding(len(self.copaal_veracity_score_valid1), 1)

        self.entity_embeddings.load_state_dict(
            {'weight': torch.tensor(
                self.convrt_embeddings(num_entities, self.embedding_dim, self.ent_embeddings))})

        self.relation_embeddings.load_state_dict(
            {'weight': torch.tensor(
                self.convrt_embeddings(num_relations, self.embedding_dim_rel, self.rel_embeddings))})

        self.sentence_embeddings_train.load_state_dict(
            {'weight': torch.tensor(
                self.convrt_embeddings(len(self.sen_embeddings_train), self.sentence_dim, self.sen_embeddings_train))})

        self.sentence_embeddings_test.load_state_dict(
            {'weight': torch.tensor(
                self.convrt_embeddings(len(self.sen_embeddings_test), self.sentence_dim, self.sen_embeddings_test))})
        self.sentence_embeddings_valid.load_state_dict(
            {'weight': torch.tensor(
                self.convrt_embeddings(len(self.sen_embeddings_valid), self.sentence_dim, self.sen_embeddings_valid))})




        self.copaal_veracity_score_train.load_state_dict(
            {'weight': torch.tensor(
                self.convrt_embeddings(len(self.copaal_veracity_score_train1), 1, self.copaal_veracity_score_train1))})

        self.copaal_veracity_score_test.load_state_dict(
            {'weight': torch.tensor(
                self.convrt_embeddings(len(self.copaal_veracity_score_test1), 1, self.copaal_veracity_score_test1))})
        self.copaal_veracity_score_valid.load_state_dict(
            {'weight': torch.tensor(
                self.convrt_embeddings(len(self.copaal_veracity_score_valid1), 1, self.copaal_veracity_score_valid1))})

        self.entity_embeddings.weight.requires_grad = False
        self.relation_embeddings.weight.requires_grad = False
        self.sentence_embeddings_test.weight.requires_grad = False
        self.sentence_embeddings_valid.weight.requires_grad = False
        self.sentence_embeddings_train.weight.requires_grad = False

        self.copaal_veracity_score_train.weight.requires_grad = False
        self.copaal_veracity_score_test.weight.requires_grad = False
        self.copaal_veracity_score_valid.weight.requires_grad = False


        self.kg_classification_layer = nn.Sequential(
            torch.nn.Linear(self.embedding_dim * 2 + self.embedding_dim_rel, self.shallom_width),
            nn.BatchNorm1d(self.shallom_width),
            nn.ReLU(),
            nn.Dropout(0.50),
            torch.nn.Linear(self.shallom_width, self.shallom_width))

        self.sentence_classification_layer = nn.Sequential(torch.nn.Linear(self.sentence_dim, self.shallom_width),
                                                           nn.BatchNorm1d(self.shallom_width),
                                                           nn.ReLU(),
                                                           nn.Dropout(0.50),
                                                           torch.nn.Linear(self.shallom_width, self.shallom_width))

        self.final_classification_layer = nn.Sequential(
            torch.nn.Linear((self.shallom_width * 2) + 1, self.shallom_width),
            nn.BatchNorm1d(self.shallom_width),
            nn.ReLU(),
            nn.Dropout(0.50),
            torch.nn.Linear(self.shallom_width, 1))

    def convrt_embeddings(self,num_entities,embedding_dim,embeddings):
        weights_matrix = np.zeros((num_entities, embedding_dim))
        words_found = 0
        for i, word in enumerate(embeddings):
            try:
                if len(word)==2305:
                    weights_matrix[i] = word[:-1]
                else:
                    weights_matrix[i] = word
                words_found += 1
            except KeyError:
                print('test')
                exit(1)
        return weights_matrix

    def forward(self, e1_idx, rel_idx, e2_idx,sen_idx="",  type="training"):
        emb_head_real = self.entity_embeddings(e1_idx)
        emb_rel_real = self.relation_embeddings(rel_idx)
        emb_tail_real = self.entity_embeddings(e2_idx)
        x = torch.cat([emb_head_real, emb_rel_real, emb_tail_real], 1)
        triplet_embedding = self.kg_classification_layer(x)
        emb_sen = []
        ver_score = 0.0
        if type.__contains__("training"):
            emb_sen = self.sentence_embeddings_train(sen_idx)
            ver_score = self.copaal_veracity_score_train(sen_idx)
        elif type.__contains__("valid"):
            emb_sen = self.sentence_embeddings_valid(sen_idx)
            ver_score = self.copaal_veracity_score_valid(sen_idx)
        else:
            emb_sen = self.sentence_embeddings_test(sen_idx)
            ver_score = self.copaal_veracity_score_test(sen_idx)
        sentence_embedding = self.sentence_classification_layer(emb_sen)

        z = torch.cat([triplet_embedding, sentence_embedding, ver_score], 1)
        return torch.sigmoid(self.final_classification_layer(z))


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

        for i, word in enumerate(self.ent_embeddings):
            self.embedding_dim = len(word)
            break
        for i, word in enumerate(self.rel_embeddings):
            self.embedding_dim_rel = len(word)
            break

        self.shallom_width = int(25.6 * self.embedding_dim)
        self.shallom_width2 = int(12.8 * self.embedding_dim)
        self.shallom_width3 = int(1 * self.embedding_dim)
        self.sen_embeddings_train =  dataset.emb_sentences_train
        self.sen_embeddings_test = dataset.emb_sentences_test
        self.sen_embeddings_valid = dataset.emb_sentences_valid

        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim_rel )
        self.sentence_embeddings_train = nn.Embedding(len(self.sen_embeddings_train), self.sentence_dim)
        self.sentence_embeddings_test = nn.Embedding(len(self.sen_embeddings_test), self.sentence_dim)
        self.sentence_embeddings_valid = nn.Embedding(len(self.sen_embeddings_valid), self.sentence_dim)

        self.entity_embeddings.load_state_dict(
            {'weight': torch.tensor(self.convrt_embeddings(num_entities, embedding_dim, self.ent_embeddings))})

        self.relation_embeddings.load_state_dict(
            {'weight': torch.tensor(self.convrt_embeddings(num_relations, self.embedding_dim_rel, self.rel_embeddings))})

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

        self.kg_classification_layer = nn.Sequential(torch.nn.Linear(self.embedding_dim * 2 + self.embedding_dim_rel , self.shallom_width),
                                                     nn.BatchNorm1d(self.shallom_width),
                                                     nn.ReLU(),
                                                     nn.Dropout(0.50),
                                                     torch.nn.Linear(self.shallom_width, self.shallom_width))

        self.sentence_classification_layer = nn.Sequential(torch.nn.Linear(self.sentence_dim, self.shallom_width),
                                                           nn.BatchNorm1d(self.shallom_width),
                                                           nn.ReLU(),
                                                           nn.Dropout(0.50),
                                                           torch.nn.Linear(self.shallom_width, self.shallom_width))

        self.final_classification_layer = nn.Sequential(torch.nn.Linear(self.shallom_width * 2, self.shallom_width),
                                                        nn.BatchNorm1d(self.shallom_width),
                                                        nn.ReLU(),
                                                        nn.Dropout(0.50),
                                                        torch.nn.Linear(self.shallom_width, 1))

    def convrt_embeddings(self,num_entities,embedding_dim,embeddings):
        weights_matrix = np.zeros((num_entities, embedding_dim))
        words_found = 0
        for i, word in enumerate(embeddings):
            try:
                if len(word) == 2305:
                    weights_matrix[i] = word[:-1]
                else:
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
        triplet_embedding = self.kg_classification_layer(x)
        emb_sen =[]
        if type.__contains__("training"):
            emb_sen = self.sentence_embeddings_train(sen_idx)
        elif type.__contains__("valid"):
            emb_sen = self.sentence_embeddings_valid(sen_idx)
        else:
            emb_sen = self.sentence_embeddings_test(sen_idx)
        sentence_embedding = self.sentence_classification_layer(emb_sen)
        z = torch.cat([triplet_embedding,sentence_embedding],1)
        return torch.sigmoid(self.final_classification_layer(z))


from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

# input to the system
epochs = 100
emb_model = "TransE"
bpdp_ds = True
# "date/",
complete_data_set = ["complete_dataset"]
properties_split = ["deathPlace/","birthPlace/","author/","award/","foundationPlace/","spouse/","starring/","subsidiary/"]
datasets_class = ["property/","random/","range/","mix/","domain/","domainrange/"]
# make it true or false
prop_split = False
save_model = False
full_hybrid = True
# no point of taking complete dataset if the data is imbalanced
complete_data = False
clss = datasets_class
if complete_data:
    clss = complete_data_set
elif prop_split:
    clss = properties_split
elif bpdp_ds:
    clss = ["bpdp"]

for cls in clss:
    method = "full-hybrid" #emb-only  hybrid, full-hybrid
    path_dataset_folder = '../dataset/'
    if bpdp_ds==True:
        path_dataset_folder = '../dataset/data/bpdp/'
        dataset = Data(data_dir=path_dataset_folder, subpath=None, complete_data=complete_data, emb_typ=emb_model,
                   emb_file="../",bpdp_dataset=True,full_hybrid=True)
    elif complete_data:
        dataset = Data(data_dir=path_dataset_folder, subpath=None, complete_data=complete_data, emb_typ= emb_model,emb_file = "../")
    elif prop_split:
        dataset = Data(data_dir=path_dataset_folder, subpath= None, prop = cls, emb_typ= emb_model,emb_file = "../")
    elif full_hybrid:
        path_dataset_folder = '../dataset/data/copaal/'
        dataset = Data(data_dir=path_dataset_folder, subpath= cls, emb_typ= emb_model, emb_file = "../", full_hybrid=True)
    else:
        dataset = Data(data_dir=path_dataset_folder, subpath= cls, emb_typ= emb_model,emb_file = "../")
    num_entities, num_relations = len(dataset.entities), len(dataset.relations)
    if method == "emb-only":
        model = BaselineEmdeddingsOnlyModel(embedding_dim=100, num_entities=num_entities, num_relations=num_relations, dataset=dataset)
    elif method == "full-hybrid":
        model = AnaBaselineEmdeddingsOnlyModel(embedding_dim=100, num_entities=num_entities, num_relations=num_relations, dataset=dataset)
    else:
        model = HybridModel(embedding_dim=100, num_entities=num_entities, num_relations=num_relations, dataset=dataset)

    print(model)
    # writer = SummaryWriter('../dataset/logsdir')
    bat_size = int(len(dataset.idx_train_data)/3)+1
    bat_size_valid = int(len(dataset.idx_valid_data) / 2) + 1
    X_dataloader = DataLoader(torch.Tensor(dataset.idx_train_data).long(), batch_size=bat_size, num_workers=0, shuffle=True)
    X_valid_dataloader = DataLoader(torch.Tensor(dataset.idx_valid_data).long(), batch_size=bat_size_valid, num_workers=0, shuffle=False)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # accuracy                           0.63      1135

    optimizer = torch.optim.Adam(model.parameters())
    # Declaring Criterion and Optimizer  BCE is good for classification
    criterion = nn.BCELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    min_valid_loss = np.inf

    # test_jj, valid_jj = generate_data(dataset.test_data, dataset.valid_set)
    for i in range(epochs):
        jj = []
        iter=0
        loss_of_epoch = 0
        model.train()
        for fact_mini_batch in X_dataloader:

            idx_s, idx_p, idx_o, label, x_data = fact_mini_batch[:, 0], fact_mini_batch[:, 1], fact_mini_batch[:,
                                                                                       2], fact_mini_batch[:, 3], fact_mini_batch[:, 4]
            # Label conversion
            label = label.float()

            # 2. Forward pass
            pred = model(idx_s, idx_p, idx_o,x_data,"training").flatten()

            # 3. Compute Loss
            loss = criterion(pred, label)
            # Replaces pow(2.0) with abs() for L1 regularization
            l2_lambda = 0.001
            l2_norm = sum(p.abs().sum()
                          for p in model.parameters())

            loss = loss + l2_lambda * l2_norm

            # 1. Zero the gradient buffers
            optimizer.zero_grad()
            # 4. Backprop loss  Calculate gradients
            loss.backward()
            # 6. Update weights with respect to loss.
            optimizer.step()
            # Calculate Loss
            loss_of_epoch += loss.item()

        # if i % 100 == 0:
        print(f'Epoch {i + 1} \t\t Training Loss: {loss_of_epoch / len(X_dataloader)}')
        model.eval()
        iter2 = 0
        valid_loss = 0.0
        for valid_mini_batch in X_valid_dataloader:

            idx_s, idx_p, idx_o, label, x_data = valid_mini_batch[:, 0], valid_mini_batch[:, 1], valid_mini_batch[:,
                                                                                       2], valid_mini_batch[:, 3], valid_mini_batch[:, 4]
            # Label conversion
            label = label.float()

            # jj = np.arange(iter2, iter2 + len(idx_s))
            # iter2 += len(idx_s)
            # x_data = torch.tensor(jj)

            pred = model(idx_s, idx_p, idx_o, x_data, "valid").flatten()
            # Find the Loss
            loss = criterion(pred, label)
            #
            l2_lambda = 0.001
            l2_norm = sum(p.abs().sum()
                          for p in model.parameters())

            loss = loss + l2_lambda * l2_norm
            # Calculate Loss
            valid_loss += loss.item()


        print(f'Epoch {i + 1} \t\t Training Loss: {loss_of_epoch / len(X_dataloader)} \t\t Validation Loss: {valid_loss / len(X_valid_dataloader)}')
        if  min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            # if i%100==0:
            if save_model:
                torch.save(model.state_dict(),
                           path_dataset_folder+'models/'+emb_model+'-'+cls.replace("/","-")+'-' +method+'-' + str(200) + '.pth')
            # torch.save(model.state_dict(), path_dataset_folder+'models/'+cls.replace("/","-")+'saved_model.pth')

    # Train F1 train dataset
    X_train = np.array(dataset.idx_train_data)[:, :5]
    y_train = np.array(dataset.idx_train_data)[:, -2]
    X_train_tensor = torch.Tensor(X_train).long()

    # jj = np.arange(0, len(X_train))
    # x_data = torch.tensor(jj)

    # X_sen_train_tensor = torch.Tensor(X_sen_train).long()
    idx_s, idx_p, idx_o, x_data = X_train_tensor[:, 0], X_train_tensor[:, 1], X_train_tensor[:, 2], X_train_tensor[:, 4]
    prob = model(idx_s, idx_p, idx_o,x_data).flatten()
    pred = (prob > 0.5).float()
    pred = pred.data.detach().numpy()
    print('Acc score on train data', accuracy_score(y_train, pred))
    print('report:', classification_report(y_train,pred))

    # Train F1 test dataset
    X_test = np.array(dataset.idx_test_data)[:, :5]
    y_test = np.array(dataset.idx_test_data)[:, -2]
    X_test_tensor = torch.Tensor(X_test).long()
    idx_s, idx_p, idx_o, x_data = X_test_tensor[:, 0], X_test_tensor[:, 1], X_test_tensor[:, 2], X_test_tensor[:, 4]

    # jj = np.arange(0, len(X_test))
    # x_data = torch.tensor(jj)

    prob = model(idx_s, idx_p, idx_o,x_data, type="testing").flatten()
    pred = (prob > 0.5).float()
    pred = pred.data.detach().numpy()
    print('Acc score on test data', accuracy_score(y_test, pred))
    print('report:', classification_report(y_test,pred))
    exit(1)
    # torch.save(model.state_dict(), path_dataset_folder+'data/train/'+cls+'TransEmodel-'+method+'-'+str(epochs)+'.pth')


# git remote set-url origin2 https://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@github.com/factcheckerr/HybridFC.git
