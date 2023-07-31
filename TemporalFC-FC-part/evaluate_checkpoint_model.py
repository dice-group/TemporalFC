import os
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
# from writeCheckpointPredictionsInFile import save_data
from pytorch_lightning import LightningModule
from main import argparse_default
from data import Data
import torch
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score

# from nn_models import HybridModel, TransE, complex
from utils.static_funcs import calculate_wilcoxen_score, select_model

class MyLightningModule(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.load_h()


def save_data(dataset, data_dir="", training="test",  scores=[], method="hybrid"):
    # saving the ground truth values
    data = list()
    if training=="train":
        data = dataset.train_set
        folder = "train"
    elif training == "test":
        data = dataset.test_data
        folder = "test"
    elif training == "valid":
        data = dataset.valid_data
        folder = "test"
    else:
        exit(1)

    with open(data_dir +"-"+ "ground_truth_"+training+ "_"+method+".nt", "w") as prediction_file:
        new_line = "\n"
        # <http://swc2019.dice-research.org/task/dataset/s-00001> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement> .
        for idx, (head, relation, tail, time, score) in enumerate(
                (data)):
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>\t" + "<http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement>\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>\t<http://rdf.freebase.com/ns/" + \
                        head.split("/")[-1][:-1] + ">\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>\t<http://rdf.freebase.com/ns/" + \
                        relation.split("/")[-1][:-1] + ">\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>\t<http://rdf.freebase.com/ns/" + \
                        tail.split("/")[-1][:-1] + ">\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://swc2017.aksw.org/hasTruthValue>\t\"" + str(
                score) + "\"^^<http://www.w3.org/2001/XMLSchema#double>\t.\n"
        prediction_file.write(new_line)


    # saving the pridiction values
    with open(data_dir +"-"+ "prediction_"+training+ "_pred_"+method+".nt", "w") as prediction_file:
        new_line = "\n"
        for idx, (tuple, score) in enumerate(
                zip(data,scores)):
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>\t" + "<http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement>\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>\t<http://rdf.freebase.com/ns/" + \
                        tuple[0].split("/")[-1][:-1] + ">\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>\t<http://rdf.freebase.com/ns/" + \
                        tuple[1].split("/")[-1][:-1] + ">\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>\t<http://rdf.freebase.com/ns/" + \
                        tuple[2].split("/")[-1][:-1] + ">\t.\n"
            new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(
                idx) + ">\t<http://swc2017.aksw.org/hasTruthValue>\t\"" + str(
                float(score)) + "\"^^<http://www.w3.org/2001/XMLSchema#double>\t.\n"
        prediction_file.write(new_line)

    # saving the pridiction values
    with open(data_dir +"-"+ "prediction_"+training+ "_pred_"+method+".txt", "w") as prediction_file:
        new_line = "\n"
        for idx, (tuple, score) in enumerate(
                zip(data,scores)):
            new_line += "<http://dbpedia.org/resource/" + \
                        tuple[0].split("/")[-1][:-1] + ">\t"
            new_line += "<http://dbpedia.org/ontology/" + \
                        tuple[1].split("/")[-1][:-1] + ">\t"
            new_line += "<http://dbpedia.org/resource/" + \
                        tuple[2].split("/")[-1][:-1] + ">\t"
            new_line += str(float(score)) + "\"^^<http://www.w3.org/2001/XMLSchema#double>\t.\n"
        prediction_file.write(new_line)


def restore_checkpoint(self, model: "pl.LightningModule", ckpt_path: Optional[str] = None):
    return  model.load_from_checkpoint(ckpt_path)


args = argparse_default()


# for cc in clss:
    # methods = ["temporal", "full-Hybrid", "KGE-only", "text-only", "path-only", "text-KGE-Hybrid", "text-path-Hybrid", "KGE-path-Hybrid"]
    # methods = ["full-hybrid","temporal"] #["KGE-only","text-KGE-Hybrid"]
df_test = pd.DataFrame()
df_train = pd.DataFrame()
    # for cls in methods:

# method = cls #emb-only  #hybrid

        # args.path_dataset_folder = 'dataset/'
cmodel = args.model
        # args.subpath = cc
        # hybrid_app = False
        # args.path_dataset_folder = 'dataset/'
sub_dataset_path = "" if (args.sub_dataset_path==None) else args.sub_dataset_path
dataset_subset = sub_dataset_path
print("processing:" + cmodel + "--" + dataset_subset)
        # if args.model == "full-hybrid":
        #     args.path_dataset_folder += 'hybrid_data/copaal/'
        #     hybrid_app = True

if args.eval_dataset=="Dbpedia5" and ((str(args.model).lower().__contains__("full-hybrid")) or (str(args.model).lower().__contains__("path"))) :
    args.model = "text-KGE-Hybrid"
    cmodel = args.model


args.dataset = Data(args)
# args.num_entities, args.num_relations = args.dataset.num_entities, args.dataset.num_relations
args.num_entities, args.num_relations, args.num_times = args.dataset.num_entities, args.dataset.num_relations, args.dataset.num_times

model, frm = select_model(args)

dirs = os.listdir(os.path.dirname(os.path.abspath("dataset")) + "/dataset/HYBRID_Storage/")

for flder in dirs:
    chkpnts = os.listdir(os.path.dirname(os.path.abspath("dataset"))+ "/dataset/HYBRID_Storage/" + flder)
    for chk in chkpnts:
        cc_change = dataset_subset
        if dataset_subset.__contains__("/"):
            cc_change= dataset_subset[:-1].lower()
        if chk.startswith("sample") and chk.lower().__contains__(dataset_subset.replace("/", "").lower()) \
                and (chk.lower()).__contains__(cmodel.lower()) and (chk.lower()).__contains__( str(args.emb_type).lower() + "")\
                and (chk.lower()).__contains__(cc_change.lower())\
                and (chk.lower()).__contains__(str(args.negative_triple_generation).lower()):
            print("checkpoint to be evaluated: "+chk)
            file_name = chk #"sample-"+cls.replace("/","")+"=0--"+cls2.replace("/","")+"=0-epoch=09-val_loss=0.00.ckpt"
            pth = os.path.dirname(os.path.abspath(file_name)).replace("comparison","")+"/dataset/HYBRID_Storage/"+flder+"/"+file_name
            print("Resuls for " + dataset_subset)
            model = model.load_from_checkpoint(pth,args=args)

            model.eval()

            # Train F1 train dataset
            X_train = np.array(args.dataset.idx_train_data)[:, :6]
            y_train = np.array(args.dataset.idx_train_data)[:, -2]

            X_train_tensor = torch.Tensor(X_train).long()

            jj = np.arange(0, len(X_train))
            x_data = torch.tensor(jj)

            # X_sen_train_tensor = torch.Tensor(X_sen_train).long()
            idx_s, idx_p, idx_o, idx_t, x_data = X_train_tensor[:, 0], X_train_tensor[:, 1], X_train_tensor[:, 2], X_train_tensor[:, 3], X_train_tensor[:, 5]
            # idx_s, idx_p, idx_o = X_train_tensor[:, 0], X_train_tensor[:, 1], X_train_tensor[:, 2]

            prob = model.forward_triples(idx_s, idx_p, idx_o, idx_t, x_data).flatten()
            # prob = model.forward_triples(idx_s, idx_p, idx_o,x_data).flatten()
            np.savetxt(os.path.dirname(os.path.abspath("dataset")) + "/dataset/HYBRID_Storage/" + flder + "/"+'predictions_train.txt', prob.detach().numpy())
            pred = (prob > 0.50).float()
            pred = pred.data.detach().numpy()
            print('Acc score on train data', accuracy_score(y_train, pred))
            print('report:', classification_report(y_train,pred))
            df_train[cmodel] = pred

            save_data(args.dataset,
                      os.path.dirname(os.path.abspath("dataset")) + "/dataset/HYBRID_Storage/" + flder + "/" + chk.split("epoch=")[1].split("-val_loss")[0],
                      "train", pred, cmodel)

                    # Train F1 test dataset
            X_test = np.array(args.dataset.idx_test_data)[:, :6]
            y_test = np.array(args.dataset.idx_test_data)[:, -2]
            X_test_tensor = torch.Tensor(X_test).long()
            idx_s, idx_p, idx_o, idx_t, x_data = X_test_tensor[:, 0], X_test_tensor[:, 1], X_test_tensor[:, 2], X_test_tensor[:, 3], X_test_tensor[:, 5]
            # idx_s, idx_p, idx_o = X_test_tensor[:, 0], X_test_tensor[:, 1], X_test_tensor[:, 2]

            jj = np.arange(0, len(X_test))
            x_data = torch.tensor(jj)
            prob = model.forward_triples(idx_s, idx_p, idx_o, idx_t, x_data, type="test").flatten()
            # prob = model.forward_triples(idx_s, idx_p, idx_o,x_data,"testing").flatten()
            np.savetxt(os.path.dirname(os.path.abspath("dataset")) +
                       "/dataset/HYBRID_Storage/" + flder + "/" + 'predictions_test.txt', prob.detach().numpy())
            pred = (prob > 0.50).float()
            pred = pred.data.detach().numpy()
            print('Acc score on test data', accuracy_score(y_test, pred))
            print('report:', classification_report(y_test,pred))
            df_test[cmodel] = pred
            print("Note: Deleting the checkpoints and saving the prediction files in same folder:"+str(pth))
            save_data(args.dataset,
                      os.path.dirname(os.path.abspath("dataset")) + "/dataset/HYBRID_Storage/" + flder + "/" + chk.split("epoch=")[1].split("-val_loss")[0],
                              "test", pred, cmodel)
            os.remove(os.path.dirname(os.path.abspath("dataset"))+ "/dataset/HYBRID_Storage/" + flder +"/"+chk)
            os.remove(os.path.dirname(os.path.abspath("dataset")) + "/dataset/HYBRID_Storage/" + flder + "/model.pt")
                    # exit(1)

# # stats test
# calculate_wilcoxen_score(df_train,"train")
# calculate_wilcoxen_score(df_test,"test")

