from data import Data
# from embedding_only_approach import Baseline, Baseline3
import torch.nn as nn

import random
import numpy as np
import torch




def save_data_properties_split(dataset, data_dir="", multiclass="", training="test", scores=[], method="hybrid", emb="TransE"):
    # saving the ground truth values
    data = list()
    if training == "train":
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
    properties_split = ["deathPlace/", "birthPlace/", "author/", "award/", "foundationPlace/", "spouse/", "starring/",
                        "subsidiary/"]
    for prop in properties_split:
        with open(data_dir + "data/" + folder + "/" + multiclass + emb + "/" +prop +"ground_truth_" + training + "_" + method + ".nt",
                "w") as prediction_file:
            new_line = "\n"
            # <http://swc2019.dice-research.org/task/dataset/s-00001> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement> .
            for idx, (head, relation, tail, score) in enumerate(
                    (data)):
                if relation.__contains__(prop[:-1]):
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
        with open(
                data_dir + "data/" + folder + "/" + multiclass + emb + "/" +prop + "prediction_" + training + "_pred_" + method + ".nt",
                "w") as prediction_file:
            new_line = "\n"
            for idx, (tuple, score) in enumerate(
                    zip(data, scores)):
                if tuple[1].__contains__(prop[:-1]):
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


def save_data(dataset, data_dir="", multiclass="", training="test",  scores=[], method="KV_rule"):
    # saving the ground truth values
    data = list()
    if training=="train/":
        data = dataset.train_set
        folder = "train"
    elif training == "test/":
        data = dataset.test_data
        folder = "test"
    elif training == "valid/":
        data = dataset.valid_data
        folder = "test"
    else:
        exit(1)
    data1 = []
    for s,p,o, pred in data:
        for s1,p1,o1, pred1 in scores:
            if s.__contains__(s1) and o.__contains__(o1) and p.__contains__(p1):
                data1.append((s,p,o,pred))
                break


    if len(data1)!=len(scores):
        print("sizes are different..should be same")
        exit(1)

    with open(data_dir + "ground_truth_"+folder+ "_"+method+".nt", "w") as prediction_file:
        new_line = "\n"
        # <http://swc2019.dice-research.org/task/dataset/s-00001> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement> .
        for idx, (head, relation, tail, score) in enumerate(
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
    with open(data_dir + "prediction_"+folder+ "_pred_"+method+".nt", "w") as prediction_file:
        new_line = "\n"
        for idx, (head, relation, tail, score) in enumerate(
                (scores)):
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

# "date/",
bpdp = True
data_type = ["train/","test/"]
datasets_class = ["domain/","domainrange/","mix/","property/","random/","range/"]
name = "result-scored"
# cls = 6
dataset1 = "factbench/"
path_dataset_folder = '../dataset/'

if bpdp:
    datasets_class = [""]
    dataset1 = "bpdp/"
    path_dataset_folder += "data/bpdp/"
import csv
for cls in datasets_class:

    dataset = Data(data_dir=path_dataset_folder, subpath=cls, emb_file="../",bpdp_dataset=bpdp)
    for typ in data_type:

        file_dataset_folder = '../dataset/korean/'+dataset1+typ + cls
        datas = []
        with open("%s%s.tsv" % (file_dataset_folder,name)) as file:
            tsv_file = csv.reader(file, delimiter="\t")
            for line in tsv_file:
                print(line)
                datas.append(line)

        save_data(dataset, file_dataset_folder, cls, training=typ, scores=datas)
        print("done")
