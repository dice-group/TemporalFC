from data import Data
from sklearn.metrics import auc
import numpy as np

# datasets_class = ["date/","domain/","domainrange/","mix/","property/","random/","range/"]
#
# path_dataset_folder = 'dataset/'
# dataset = Data(data_dir=path_dataset_folder, subpath= datasets_class[1])

from data import Data
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# extracting evedience sentence and generating training and testing triples from the list of facts in factbecnh.
class Baseline_Results:
    def __init__(self, data_dir=None, multiclass=True, bpdp = False, fullhybrid = False):
        # self.data = []
        if multiclass:
            datasets_class = ["date/", "domain/", "domainrange/", "mix/", "property/", "random/", "range/"]
            for sub_path in datasets_class:
                self.extract_and_save_result_of_factcheck_output_multiclass(self, data_dir, sub_path, fullhybrid)
        elif bpdp == True:
            self.extract_and_save_result_of_factcheck_output_bpdp(self, data_dir, fullhybrid)
        else:
             self.extract_and_save_result_of_factcheck_output(self, data_dir, fullhybrid)


    @staticmethod
    def extract_and_save_result_of_factcheck_output_bpdp(self, data_dir,fullhybrid):
        self.dataset = Data(data_dir=data_dir, emb_file='../', bpdp_dataset=True, full_hybrid=fullhybrid)
        print("test")
        self.save_data(self,data_dir)

    @staticmethod
    def extract_and_save_result_of_factcheck_output(self, data_dir, fullhybrid):
        self.dataset = Data(data_dir=data_dir+"complete_dataset/", full_hybrid= fullhybrid)
        print("test")
        self.save_data(self,data_dir+"complete_dataset/")




    @staticmethod
    def extract_and_save_result_of_factcheck_output_multiclass(self, data_dir, multiclass, fullhybrid):
        self.dataset = Data(data_dir=data_dir, subpath=multiclass, full_hybrid=fullhybrid, emb_file="../")
        print("test")
        self.save_data(self,data_dir,multiclass)

    @staticmethod
    def save_data(self,data_dir="",multiclass="" ):
        # saving the ground truth values
        dirr = "data/"
        if data_dir.__contains__("copaal"):
            dirr = ""
        with open(data_dir+dirr+"test/"+multiclass+"ground_truth_test.nt", "w") as prediction_file:
            new_line = "\n"
            # <http://swc2019.dice-research.org/task/dataset/s-00001> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement> .
            for idx, (head, relation, tail, score) in enumerate(
                    (self.dataset.test_data)):
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>\t" + "<http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement>\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>\t<http://rdf.freebase.com/ns/" + head.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>\t<http://rdf.freebase.com/ns/" + relation.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>\t<http://rdf.freebase.com/ns/" + tail.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://swc2017.aksw.org/hasTruthValue>\t\"" + str(score) + "\"^^<http://www.w3.org/2001/XMLSchema#double>\t.\n"
            prediction_file.write(new_line)
        with open(data_dir+dirr+"train/"+multiclass+"ground_truth_train.nt", "w") as prediction_file:
            new_line = "\n"
            for idx, (head, relation, tail, score) in enumerate(
                    (self.dataset.train_set)):
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>\t" + "<http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement>\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>\t<http://rdf.freebase.com/ns/" + head.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>\t<http://rdf.freebase.com/ns/" + relation.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>\t<http://rdf.freebase.com/ns/" + tail.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://swc2017.aksw.org/hasTruthValue>\t\"" + str(score) + "\"^^<http://www.w3.org/2001/XMLSchema#double>\t.\n"

            prediction_file.write(new_line)

        with open(data_dir+dirr+"test/"+multiclass+"ground_truth_valid.nt", "w") as prediction_file:
            new_line = "\n"
            for idx, (head, relation, tail, score) in enumerate(
                    (self.dataset.valid_data)):
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>\t" + "<http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement>\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>\t<http://rdf.freebase.com/ns/" + head.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>\t<http://rdf.freebase.com/ns/" + relation.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>\t<http://rdf.freebase.com/ns/" + tail.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://swc2017.aksw.org/hasTruthValue>\t\"" + str(score) + "\"^^<http://www.w3.org/2001/XMLSchema#double>\t.\n"

            prediction_file.write(new_line)

        # saving the pridiction values
        with open(data_dir+dirr+"test/"+multiclass+"prediction_test_pred.nt", "w") as prediction_file:
            new_line = "\n"
            for idx, (head, relation, tail, score) in enumerate(
                    (self.dataset.test_data_pred)):
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>\t" + "<http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement>\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>\t<http://rdf.freebase.com/ns/" + head.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>\t<http://rdf.freebase.com/ns/" + relation.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>\t<http://rdf.freebase.com/ns/" + tail.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://swc2017.aksw.org/hasTruthValue>\t\"" + str(score) + "\"^^<http://www.w3.org/2001/XMLSchema#double>\t.\n"
            prediction_file.write(new_line)

        with open(data_dir+dirr+"train/"+multiclass+"prediction_train_pred.nt", "w") as prediction_file:
            new_line = "\n"
            for idx, (head, relation, tail, score) in enumerate(
                    (self.dataset.train_set_pred)):
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>\t" + "<http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement>\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>\t<http://rdf.freebase.com/ns/" + head.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>\t<http://rdf.freebase.com/ns/" + relation.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>\t<http://rdf.freebase.com/ns/" + tail.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://swc2017.aksw.org/hasTruthValue>\t\"" + str(score) + "\"^^<http://www.w3.org/2001/XMLSchema#double>\t.\n"

            prediction_file.write(new_line)

        with open(data_dir + dirr+"test/" + multiclass + "prediction_valid_pred.nt", "w") as prediction_file:
            new_line = "\n"
            for idx, (head, relation, tail, score) in enumerate(
                    (self.dataset.valid_data_pred)):
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>\t" + "<http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement>\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#subject>\t<http://rdf.freebase.com/ns/" + head.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate>\t<http://rdf.freebase.com/ns/" + relation.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://www.w3.org/1999/02/22-rdf-syntax-ns#object>\t<http://rdf.freebase.com/ns/" + tail.split("/")[-1][:-1] + ">\t.\n"
                new_line += "<http://swc2019.dice-research.org/task/dataset/s-00" + str(idx) + ">\t<http://swc2017.aksw.org/hasTruthValue>\t\"" + str(score) + "\"^^<http://www.w3.org/2001/XMLSchema#double>\t.\n"

            prediction_file.write(new_line)




multiclass_factbench = False
bpdp = True
full_hybrid = False
path_dataset_folder = '../dataset/'
if bpdp == True:
    path_dataset_folder = path_dataset_folder+ 'data/bpdp/'
    multiclass_factbench = False
if full_hybrid == True:
    path_dataset_folder = path_dataset_folder+ 'data/copaal/'
se = Baseline_Results(data_dir=path_dataset_folder, multiclass=multiclass_factbench, bpdp=bpdp, fullhybrid= full_hybrid)








#

#


# dx = 5
# xx = np.arange(1,100,dx)
# yy = np.arange(1,100,dx)
#
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(xx,yy)))
# print('computed AUC using np.trapz: {}'.format(np.trapz(yy, dx = dx)))
