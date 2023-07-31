# from torch import hub
from sentence_transformers import SentenceTransformer
from urllib.parse import quote, unquote
import json
import pytorch_lightning as pl
import argparse
import os
import numpy as np
import pandas as pd
import zipfile
import logging
logging.basicConfig(level=logging.INFO)
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

'''
This class is for
extracting evedience sentence and generating embeddings 
from those evedence sentences and storing them in CSV file format
'''
class UpdateResIRIs:
    def __init__(self, data_dir=None):
        self.data_file = "result_TC.txt"
        self.k = 3
        self.extract_sentence_embeddings_from_factcheck_output_DBpedia34k_dataset(self, data_dir)

    @staticmethod
    def getCOPAAL_score(self, data_dir, fact, cat, dataset="copaal"):
        # Opening JSON file
        score = 0.0
        updated_score = False
        # if cat != "True":
        f = open(data_dir + dataset + '/' + cat + '.json')
        # returns JSON object as
        # a dictionary
        data = json.load(f)
        for dd in data['results']:
            fn = dd['filename']
            # if not fn.__contains__(fact):
            #     exit(1)
            if fn.__contains__(fact):
                print(fn)
                score = dd['result']['veracityValue']
                updated_score = True
                break

        # else:
        #     f = open(data_dir + 'copaal/correct.json')
        #     data = json.load(f)
        #     for dd in data['results']:
        #         fn = dd['filename']
        #         # if not fn.__contains__(fact):
        #         #     exit(1)
        #         if fn.__contains__(fact):
        #             print(fn)
        #             score = dd['result']['veracityValue']
        #             updated_score = True

        #
        if updated_score == False:
            print("problem detected")
            exit(1)
        return score

    @staticmethod
    def extract_sentence_embeddings_from_factcheck_output_DBpedia34k_dataset(self, data_dir=None):
        data_train = []
        data_test = []
        print(data_dir)
        ds_types = ["positive_train/", "negative_train/", "positive_test/", "negative_test/"]
        # ds_types = ["positive_test/", "negative_test/"]
        for typ in ds_types:
            data_path = data_dir + typ
            isExist = os.path.exists(data_dir + typ)
            if isExist:
                triples = set()
                with open(data_path + typ.replace("/", ".txt"), "r") as file1:
                    for line in file1:
                        print(line)
                        datapoints = line.split()
                        triples.add(datapoints[0]+"\t"+datapoints[1]+"\t"+datapoints[2])

                counter =0
                with open(data_path+ self.data_file, "r") as file1:
                    for line in file1:
                        if line.startswith("{\"taskid\":\""):
                            print(line)
                            defacto_scores = line.split("\"defactoScore\":")[1].split(",\"complexProofs\":")[0]
                            sub = unquote(line.split("\"subject\":\"")[1].split("\",\"predicate\":\"")[0])
                            pred = unquote(line.split("\"predicate\":\"")[1].split("\",\"object\":\"")[0])
                            obj = unquote(line.split("\"object\":\"")[1].split("\",\"file\":")[0])

                            new_triple = next(iter(triples)).split("\t")
                            if ((sub == new_triple[0] and pred == new_triple[1]) or
                                (pred == new_triple[1] and obj == new_triple[2])):
                                print("ok")
                            else:
                                print("problem")

                            counter+=1

                            top_three_sentences = []
                            top_three_sentences_trust_scores = []
                            if sub== None or obj == None or pred == None:
                                print("error")
                                exit(1)
                            complex_proof =  line.split("\"complexProofs\":[")[1].split("],\"subject")[0]
                            if complex_proof!="":
                                print("it contains the proof")
                                proofPhrases = complex_proof.split("{\"website\":")
                                evidence_sentence_pg_rank = dict()
                                evidence_sentence_trust = dict()
                                for phrase in proofPhrases[1:]:
                                    proofPhrase = phrase.split("proofPhrase\":\"")[1].split("\"}")[0]
                                    pagerank_score = phrase.split("pagerank\":")[1].split(",\"")[0]
                                    trustworthiness_score = phrase.split("trustworthiness\":")[1].split(",\"")[0]
                                    print(proofPhrase)
                                    print(pagerank_score)
                                    print(trustworthiness_score)
                                    evidence_sentence_pg_rank[proofPhrase] = float(pagerank_score)
                                    evidence_sentence_trust[proofPhrase] = float(trustworthiness_score)

                                sorted_pg_ranks = sorted(evidence_sentence_pg_rank.values(), reverse=True)
                                for pg_rank in sorted_pg_ranks:
                                    for sent in evidence_sentence_pg_rank.keys():
                                        if evidence_sentence_pg_rank[sent] == pg_rank:
                                            top_three_sentences.append(sent)
                                            top_three_sentences_trust_scores.append(evidence_sentence_trust[sent])
                                        if len(top_three_sentences) >= self.k:
                                            break
                                    if len(top_three_sentences) >= self.k:
                                        break
                                print("final sentences -------")
                                print(top_three_sentences)
                                # model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
                            if typ.__contains__("positive"):
                                correct = True
                            else:
                                correct = False
                            if typ.__contains__("test"):
                                print("test" + str(correct))
                                data_test.append(
                                    [sub, pred, obj, correct, top_three_sentences, top_three_sentences_trust_scores,
                                     defacto_scores])

                            if typ.__contains__("train"):
                                print("train" + str(correct))
                                data_train.append(
                                    [sub, pred, obj, correct, top_three_sentences, top_three_sentences_trust_scores,
                                     defacto_scores])
            else:
                print(typ+" folder does not exists")

        model = SentenceTransformer('sentence-transformers/all-distilroberta-v1',device='cpu')
        true_statements_embeddings = {}
        n = 3
        embeddings_textual_evedences_train_date, true_statements_embeddings = self.getSentenceEmbeddings(self, data_dir,
                                                                                                         data_test,
                                                                                                         model,
                                                                                                         true_statements_embeddings,
                                                                                                         "test", n,
                                                                                                         False, bpdp=False)
        true_statements_embeddings_test = {}
        embeddings_textual_evedences_test_date, true_statements_embeddings_test = self.getSentenceEmbeddings(self,
                                                                                                             data_dir,
                                                                                                             data_train,
                                                                                                             model,
                                                                                                             true_statements_embeddings_test,
                                                                                                             "train",
                                                                                                             n, False, bpdp=True)

        exit(1)

    # path of the training or testing folder
    # data2 contains data
    # type2 contains either test or train string
    @staticmethod
    def saveSentenceEmbeddingToFile(self, path, data2 , type2,n):
        X = np.array(data2)
        print(X.shape)
        # X = X.reshape(X.shape[0], 768*n)
        header = []
        vals = []
        for test in data2:
            header.append(np.concatenate((list(test.keys())[0].replace(',','').split(' '),(np.array(list(test.values()))).flatten()), axis=0))
            # header.append(test.values())

        X = pd.DataFrame(header)
        compression_opts = dict(method='zip', archive_name=type2+'SE.csv')
        X.to_csv(path + type2+'SE.zip', index=False, compression=compression_opts)
        with zipfile.ZipFile(path +type2+ 'SE.zip', 'r') as zip_ref:
            zip_ref.extractall(path)
        print("data saved")




    @staticmethod
    def getSentenceEmbeddings(self, data_dir, data, model, true_statements_embeddings, type = 'default', n =3, props = False, bpdp = False):
        embeddings_textual_evedences = []
        for idx, (s, p, o, c, p2, t1, s1) in enumerate(data):
            print('triple->'+s+' '+p+' '+o)
            p3 = p2.copy()
            sentence_embeddings = dict()

            if (s + ' ' + p + ' ' + o) not in true_statements_embeddings.keys():
                sentence_embeddings[s + ' ' + p + ' ' + o] = model.encode(p3)
                true_statements_embeddings[s + ' ' + p + ' ' + o] = sentence_embeddings[s + ' ' + p + ' ' + o]
            else:
                sentence_embeddings[s + ' ' + p + ' ' + o] = true_statements_embeddings[s + ' ' + p + ' ' + o]

            if (np.size(sentence_embeddings[s + ' ' + p + ' ' + o]) == 0):
                sentence_embeddings[s + ' ' + p + ' ' + o] = np.zeros((n, 768), dtype=int)

            if (np.size(sentence_embeddings[s + ' ' + p + ' ' + o]) == 768 * (n - 2)):
                sentence_embeddings[s + ' ' + p + ' ' + o] = np.append(sentence_embeddings[s + ' ' + p + ' ' + o],
                                                                       (np.zeros((n - 1, 768), dtype=int)), axis=0)

            if (np.size(sentence_embeddings[s + ' ' + p + ' ' + o]) == 768 * (n - 1)):
                sentence_embeddings[s + ' ' + p + ' ' + o] = np.append(sentence_embeddings[s + ' ' + p + ' ' + o],
                                                                       (np.zeros((n - 2, 768), dtype=int)), axis=0)
            sentence_embeddings[s + ' ' + p + ' ' + o] = np.append(sentence_embeddings[s + ' ' + p + ' ' + o], s1)
            embeddings_textual_evedences.append(sentence_embeddings)


        # if type.__contains__("test"):
        path = ""
        if type.__contains__("test"):
            path = data_dir + "data/test/"
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
                print("The new directory is created!")
            self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences, "test", n)
        else:
            path = data_dir + "data/train/"
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
                print("The new directory is created!")
            self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences, "train", n)

        return embeddings_textual_evedences, true_statements_embeddings



    @staticmethod
    def getVeracityScore(self, data_dir, sub, pred, obj):
        ver = 0
        print("extracting veracity score")
        with open(data_dir +"_pred.txt", "r") as file1:
            for line in file1:
                if line.__contains__(sub+"\t"+pred+"\t"+obj):
                    ver = line.split("\t")[-1][:-1]
                    break
        if ver == 0:
            exit(1)
        return ver

def argparse_default(description=None):
    parser = pl.Trainer.add_argparse_args(argparse.ArgumentParser())
    # Paths.
    parser.add_argument("--path_dataset_folder", type=str, default='../dataset/complete_dataset/dbpedia34k/')
    parser.add_argument("--eval_dataset", type=str, default='Dbpedia34k',
                        help="Available datasets: FactBench, BPDP, Dbpedia34k")
    if description is None:
        return parser.parse_args()
    else:
        return parser.parse_args(description)

if __name__ == '__main__':
    args = argparse_default()

    # path_dataset_folder = '../dataset/'
    dataset_folder = '../dataset/complete_dataset/dbpedia34k/'

    if args.eval_dataset == "Dbpedia34k":
        se = UpdateResIRIs(data_dir=args.path_dataset_folder)

