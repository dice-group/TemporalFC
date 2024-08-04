# TODO  using existing model to perform fact checking task!
import os
from typing import Optional
from sklearn.metrics import classification_report

import pandas as pd
import pytorch_lightning as pl
from comparison.writeCheckpointPredictionsInFile import save_data
from pytorch_lightning import LightningModule
from main import argparse_default
from data_TP import Data
import torch
from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score

from utils_TP.static_funcs import calculate_wilcoxen_score, select_model
import argparse

class MyLightningModule(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.load_h()


def hit_at_k(prediction, target, k):
    # Sort the predictions in descending order
    sorted_prediction = prediction

    # Get the top k predictions
    top_k = sorted_prediction[:k]
    top_k = top_k
    # Check if the target label is in the top k predictions
    hit = target in top_k

    return hit
def hits_k(predictions, labels,k):

    # Convert predictions and labels to numpy arrays
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Compute the reciprocal rank for each query
    hits = []
    for query_index in range(len(predictions)):
        # Get the prediction and label for the current query
        prediction = predictions[query_index]
        label = labels[query_index]
        hit = hit_at_k (prediction,label,k)
        hits.append(hit)
        # Return the mean of all the reciprocal ranks
    return np.mean(hits)
def mrr_score2( predictions, labels):
        # Convert predictions and labels to numpy arrays
    predictions = np.array(predictions)
    labels = np.array(labels)

        # Compute the reciprocal rank for each query
    reciprocal_ranks = []
    for query_index in range(len(predictions)):
            # Get the prediction and label for the current query
        prediction = predictions[query_index]
        label = labels[query_index]

        # Find the rank of the highest ranked relevant item
        rank = np.where(prediction == label)[0][0] + 1
        reciprocal_rank = 1.0 / rank
        reciprocal_ranks.append(reciprocal_rank)

        # Return the mean of all the reciprocal ranks
    return np.mean(reciprocal_ranks)
def metric_measures( prob, label):
    pred = prob.data.detach().numpy()
    # max_pred = np.argmax(pred, axis=1)
    # sort_pred, idx  = torch.sort(prob, dim=1, descending=True)

    # test_mrr = mrr_score2(idx, label)
    # print(test_mrr)

    # hit_1 = hits_k(idx, label, 1)
    # hit_10 = hits_k(idx, label, 3)
    # print(hit_1)
    # print(hit_10)
    # print(label.astype(float))
    # print(pred.reshape(-1, 1) )
    threshold = 0.5
    binary_predictions = (pred >= threshold).astype(int)

    # print(accuracy_score(binary_predictions , label))

def restore_checkpoint(self, model: "pl.LightningModule", ckpt_path: Optional[str] = None):
    return  model.load_from_checkpoint(ckpt_path)

# args = argparse_default()

# datasets_class = ["domain/","domainrange/","mix/","property/","random/","range/"]
# cls2 = datasets_class[2]





def start_process():
    # properties_split = ["deathPlace/","birthPlace/","author/","award/","foundationPlace/","spouse/","starring/","subsidiary/"]
    # make it true or false
    prop_split = False
    bpdp_ds = False
    args = argparse_default()
    # args.path_dataset_folder = 'data_TP/'
    dataset = [args.eval_dataset]
    args.subpath = None


    for cc in dataset:
        # methods = ["hybridfc-full-hybrid", "KGE-only", "text-only", "path-only", "text-KGE-Hybrid", "text-path-Hybrid", "KGE-path-Hybrid","temporal-model"]
        methods = [args.model]
        for cls in methods:
            print("processing:" + cls + "--" + cc)
            method = cls #emb-only  #hybrid

            # args.path_dataset_folder = 'dataset/'
            args.model = cls
            # args.subpath = cc
            # hybrid_app = False
            # args.path_dataset_folder = 'dataset/'
            args.subpath = cc


            args.dataset = Data(args)
            args.num_entities, args.num_relations, args.num_times = args.dataset.num_entities, args.dataset.num_relations, args.dataset.num_times

            model, frm = select_model(args)

            dirs = os.listdir(os.path.dirname(os.path.abspath(args.checkpoint_dataset_folder)) + "/"+args.checkpoint_dataset_folder+"HYBRID_Storage/")
            negative_triple_type = args.negative_triple_generation
            for flder in dirs:
                if args.checkpoint_dir_folder != 'all':
                    flder = args.checkpoint_dir_folder
                chkpnts = os.listdir(os.path.dirname(os.path.abspath(args.checkpoint_dataset_folder)) + "/"+args.checkpoint_dataset_folder+"HYBRID_Storage/" + flder)
                for chk in chkpnts:
                    cc_change = cc
                    if cc.__contains__("/"):
                        cc_change = cc[:-1].lower()
                    if (chk.startswith("sample")
                            and (chk.lower()).__contains__(cls.lower())
                            and (chk.lower()).__contains__("--" + str(args.emb_type).lower() + "") \
                            and (chk.lower()).__contains__(cc_change.lower())\
                            and (chk.lower()).__contains__(negative_triple_type.lower())):
                        print(chk)
                        if (not chk.lower().__contains__('temporal-prediction')):
                            file_name = chk  # "sample-"+cls.replace("/","")+"=0--"+cls2.replace("/","")+"=0-epoch=09-val_loss=0.00.ckpt"
                            pth = os.path.dirname(os.path.abspath(file_name)).replace("comparison",
                                                                                      "") + "/dataset/HYBRID_Storage/" + flder + "/" + file_name
                            # print("Resuls for " + cc)
                            model = model.load_from_checkpoint(pth, args=args)

                            model.eval()

                            # Train F1 train dataset
                            X_train = np.array(args.dataset.idx_train_set)[:, :6]
                            y_train = np.array(args.dataset.idx_train_set)[:, -1]

                            X_train_tensor = torch.Tensor(X_train).long()

                            jj = np.arange(0, len(X_train))
                            x_data = torch.tensor(jj)
                            now = datetime.now()

                            # X_sen_train_tensor = torch.Tensor(X_sen_train).long()
                            idx_s, idx_p, idx_o, idx_t, idx_sen, idx_v = X_train_tensor[:, 0], X_train_tensor[:, 1], X_train_tensor[:, 2], X_train_tensor[:, 3], X_train_tensor[:, 4], X_train_tensor[:, 5]
                            prob = model.forward_triples(idx_s, idx_p, idx_o, idx_t, idx_sen, idx_v)
                            np.savetxt(os.path.dirname(os.path.abspath("dataset")) + "/dataset/HYBRID_Storage/" + flder + "/"+'predictions_train.txt', prob.detach().numpy())
                            # pred = (prob > 0.50).float()
                            last = datetime.now()

                            pred = prob.data.detach().numpy()
                            metric_measures(prob, label=y_train)
                            print("time of single triple: "+str(last-now))

                            # print('Acc score on train data', accuracy_score(y_train, pred))
                            # print('report:', classification_report(y_train,prob.detach().numpy()))
                            # df_train[cls] = pred
                            #
                            save_data(args.dataset,
                                      os.path.dirname(os.path.abspath(args.checkpoint_dataset_folder)) + "/"+args.checkpoint_dataset_folder+"HYBRID_Storage/" + flder + "/"+chk.split("epoch=")[1].split("-val_loss")[0],
                                      "train", pred, method)

                            # Train F1 test dataset
                            X_test = np.array(args.dataset.idx_test_set)[:, :6]
                            y_test = np.array(args.dataset.idx_test_set)[:, -1]
                            X_test_tensor = torch.Tensor(X_test).long()
                            idx_s, idx_p, idx_o, idx_t, idx_sen, idx_v = X_test_tensor[:, 0], X_test_tensor[:, 1], X_test_tensor[:, 2], X_test_tensor[:, 3], X_test_tensor[:, 4], X_test_tensor[:, 5]

                            jj = np.arange(0, len(X_test))
                            x_data = torch.tensor(jj)

                            prob = model.forward_triples(idx_s, idx_p, idx_o, idx_t, idx_sen, idx_v, "testing")
                            pred = prob.data.detach().numpy()

                            save_data(args.dataset,
                                      os.path.dirname(
                                          os.path.abspath("dataset")) + "/dataset/HYBRID_Storage/" + flder + "/" +
                                      chk.split("epoch=")[1].split("-val_loss")[0],
                                      "test", pred, method)

                            np.savetxt(os.path.dirname(os.path.abspath(
                                "dataset")) + "/dataset/HYBRID_Storage/" + flder + "/" + 'predictions_test.txt', prob.detach().numpy())
                            metric_measures(prob, label=y_test)

                            exit(1)


if __name__ == "__main__":
    start_process()