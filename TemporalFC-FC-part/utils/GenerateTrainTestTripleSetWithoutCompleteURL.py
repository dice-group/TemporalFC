from data import Data
import numpy as np
import pandas as pd
# from sentence_transformers import SentenceTransformer
# FIRST STEP
# extracting evedience sentence and generating training and testing triples from the list of facts in factbecnh.
class GenerateTrainTestTriplesSet:
    def __init__(self, data_dir=None,multiclass=True):
        self.dataset_file= "textResults4.txt"
        if multiclass:
            self.extract_sentence_embeddings_from_factcheck_output_multiclass(self,data_dir)
        else:
            self.extract_sentence_embeddings_from_factcheck_output(self,data_dir)


    @staticmethod
    def extract_sentence_embeddings_from_factcheck_output(self,data_dir):
        data_train = []
        data_test = []
        multiclass_neg_exp = True
        pred = []
        test = False
        train = False
        correct = False

        with open(data_dir+ self.dataset_file, "r") as file1:
            for line in file1:
                if line.__contains__("factbench/factbench/test/"):
                    test = True
                    train = False
                    if line.__contains__("/wrong/"):
                        correct = False
                    else:
                        correct = True
                    continue
                if line.__contains__("factbench/factbench/train/"):
                    test = False
                    train = True
                    if line.__contains__("/wrong/"):
                        correct = False
                    else:
                        correct = True

                    continue

                if line.startswith(' defactoScore'):
                    l1 = line.split("defactoScore: ")[1]
                    score = l1.split(" setProofSentences")[0]
                    x = line.split("subject : ")[1]
                    so = x.split(" object : ")
                    s = so[0].replace(" ", "_")
                    print(s)
                    assert s != ""
                    o = so[1].split(" predicate ")[0].replace(" ", "_")
                    print(o)
                    if o == "": o = "DUMMY"
                    assert o != ""
                    p = so[1].split(" predicate ")[1].replace("\n", "")
                    print(p)
                    assert p != ""
                    print("line:" + line + ":" + score + ":" + str(correct))

                    if test == True and train == False:
                        print("test")
                        data_test.append([s, p, o, correct])

                    if test == False and train == True:
                        print("train")
                        data_train.append([s, p, o, correct])

        with open(data_dir+ "train.txt", "w") as prediction_file:
            new_line = ""
            for idx, (head, relation, tail, lbl) in enumerate(data_train):
                new_line += head + "\t" + relation + "\t" + tail + "\t" + str(lbl) + "\n"

            prediction_file.write(new_line)

        with open(data_dir+"test.txt", "w") as prediction_file:
            new_line = ""
            for idx, (head, relation, tail, lbl) in enumerate(data_test):
                new_line += head + "\t" + relation + "\t" + tail + "\t" + str(lbl) + "\n"

            prediction_file.write(new_line)




    @staticmethod
    def extract_sentence_embeddings_from_factcheck_output_multiclass(self, data_dir):
        data_train = []
        data_test = []
        date_data_train = []
        date_data_test = []
        domain_data_train = []
        domain_data_test = []
        domainrange_data_train = []
        domainrange_data_test = []
        mix_data_train = []
        mix_data_test = []
        property_data_train = []
        property_data_test = []
        random_data_train = []
        random_data_test = []
        range_data_train = []
        range_data_test = []

        data_train1 = []
        data_test1 = []
        date_data_train1 = []
        date_data_test1 = []
        domain_data_train1 = []
        domain_data_test1 = []
        domainrange_data_train1 = []
        domainrange_data_test1 = []
        mix_data_train1 = []
        mix_data_test1 = []
        property_data_train1 = []
        property_data_test1 = []
        random_data_train1 = []
        random_data_test1 = []
        range_data_train1 = []
        range_data_test1 = []

        multiclass_neg_exp = True
        neg_data_dir = "../dataset/"
        pred = []
        test = False
        train = False
        correct = False

        with open(data_dir+self.dataset_file, "r") as file1:
            for line in file1:
                if line.__contains__("factbench/factbench/test/"):
                    test = True
                    train = False
                    if line.__contains__("/wrong/"):
                        correct = False
                    else:
                        correct = True
                    neg_data_dir = "true"
                    if multiclass_neg_exp:
                        if line.__contains__("/test/wrong/date"):
                            neg_data_dir = "wrong/date/"
                        if line.__contains__("/test/wrong/domain"):
                            neg_data_dir = "wrong/domain/"
                        if line.__contains__("/test/wrong/domainrange"):
                            neg_data_dir = "wrong/domainrange/"
                        if line.__contains__("/test/wrong/mix"):
                            neg_data_dir = "wrong/mix/"
                        if line.__contains__("/test/wrong/property"):
                            neg_data_dir = "wrong/property/"
                        if line.__contains__("/test/wrong/random"):
                            neg_data_dir = "wrong/random/"
                        if line.__contains__("/test/wrong/range"):
                            neg_data_dir = "wrong/range/"

                    continue
                if line.__contains__("factbench/factbench/train/"):
                    test = False
                    train = True
                    if line.__contains__("/wrong/"):
                        correct = False
                    else:
                        correct = True
                    neg_data_dir = "true"
                    if multiclass_neg_exp:
                        if line.__contains__("/train/wrong/date"):
                            neg_data_dir = "wrong/date/"
                        if line.__contains__("/train/wrong/domain"):
                            neg_data_dir = "wrong/domain/"
                        if line.__contains__("/train/wrong/domainrange"):
                            neg_data_dir = "wrong/domainrange/"
                        if line.__contains__("/train/wrong/mix"):
                            neg_data_dir = "wrong/mix/"
                        if line.__contains__("/train/wrong/property"):
                            neg_data_dir = "wrong/property/"
                        if line.__contains__("/train/wrong/random"):
                            neg_data_dir = "wrong/random/"
                        if line.__contains__("/train/wrong/range"):
                            neg_data_dir = "wrong/range/"

                    continue

                if line.startswith(' defactoScore'):
                    l1 = line.split("defactoScore: ")[1]
                    score = l1.split(" setProofSentences")[0]
                    x = line.split("subject : ")[1]
                    so = x.split(" object : ")
                    s = so[0].replace(" ", "_")
                    print(s)
                    assert s != ""
                    o = so[1].split(" predicate ")[0].replace(" ", "_")
                    print(o)
                    if o == "": o = "DUMMY"
                    assert o != ""
                    p = so[1].split(" predicate ")[1].replace("\n", "")
                    print(p)
                    assert p != ""
                    print("line:" + line + ":" + score + ":" + str(correct))

                    if test == True and train == False:
                        print("test")
                        data_test.append([s, p, o, correct])
                        data_test1.append([s, p, o, score])

                    if test == False and train == True:
                        print("train")
                        data_train.append([s, p, o, correct])
                        data_train1.append([s, p, o, score])

                    if multiclass_neg_exp == True:
                        if test == False and train == True:
                            if neg_data_dir.__contains__("/date/") or neg_data_dir.__contains__("true"):
                                date_data_train.append([s, p, o, correct])
                                date_data_train1.append([s, p, o, score])
                            if neg_data_dir.__contains__("/domain/") or neg_data_dir.__contains__("true"):
                                domain_data_train.append([s, p, o, correct])
                                domain_data_train1.append([s, p, o, score])
                            if neg_data_dir.__contains__("/domainrange/") or neg_data_dir.__contains__("true"):
                                domainrange_data_train.append([s, p, o, correct])
                                domainrange_data_train1.append([s, p, o, score])
                            if neg_data_dir.__contains__("/mix/") or neg_data_dir.__contains__("true"):
                                mix_data_train.append([s, p, o, correct])
                                mix_data_train1.append([s, p, o, score])
                            if neg_data_dir.__contains__("/property/") or neg_data_dir.__contains__("true"):
                                property_data_train.append([s, p, o, correct])
                                property_data_train1.append([s, p, o, score])
                            if neg_data_dir.__contains__("/random/") or neg_data_dir.__contains__("true"):
                                random_data_train.append([s, p, o, correct])
                                random_data_train1.append([s, p, o, score])
                            if neg_data_dir.__contains__("/range/") or neg_data_dir.__contains__("true"):
                                range_data_train.append([s, p, o, correct])
                                range_data_train1.append([s, p, o, score])

                        if test == True and train == False:
                            if neg_data_dir.__contains__("/date/") or neg_data_dir.__contains__("true"):
                                date_data_test.append([s, p, o, correct])
                                date_data_test1.append([s, p, o, score])
                            if neg_data_dir.__contains__("/domain/") or neg_data_dir.__contains__("true"):
                                domain_data_test.append([s, p, o, correct])
                                domain_data_test1.append([s, p, o, score])
                            if neg_data_dir.__contains__("/domainrange/") or neg_data_dir.__contains__("true"):
                                domainrange_data_test.append([s, p, o, correct])
                                domainrange_data_test1.append([s, p, o, score])
                            if neg_data_dir.__contains__("/mix/") or neg_data_dir.__contains__("true"):
                                mix_data_test.append([s, p, o, correct])
                                mix_data_test1.append([s, p, o, score])
                            if neg_data_dir.__contains__("/property/") or neg_data_dir.__contains__("true"):
                                property_data_test.append([s, p, o, correct])
                                property_data_test1.append([s, p, o, score])
                            if neg_data_dir.__contains__("/random/") or neg_data_dir.__contains__("true"):
                                random_data_test.append([s, p, o, correct])
                                random_data_test1.append([s, p, o, score])
                            if neg_data_dir.__contains__("/range/") or neg_data_dir.__contains__("true"):
                                range_data_test.append([s, p, o, correct])
                                range_data_test1.append([s, p, o, score])

        data_type1 = ["train","test"]
        for idx, test1 in enumerate(data_type1):
            neg_data_dir = "data/"+test1+"/date/"
            with open(data_dir+neg_data_dir+ test1+".txt", "w") as prediction_file:
                new_line = ""
                if test1 == "train":
                    data_now = date_data_train
                else:
                    data_now = date_data_test
                for idx, (head, relation, tail, lbl) in enumerate(data_now):
                    new_line += head + "\t" + relation + "\t" + tail + "\t" + str(lbl) + "\n"

                prediction_file.write(new_line)
            with open(data_dir+neg_data_dir+ test1+"_pred.txt", "w") as prediction_file:
                new_line = ""
                if test1 == "train":
                    data_now = date_data_train1
                else:
                    data_now = date_data_test1
                for idx, (head, relation, tail, score) in enumerate(data_now):
                    new_line += head + "\t" + relation + "\t" + tail + "\t" + str(score) + "\n"

                prediction_file.write(new_line)

            neg_data_dir = "data/"+test1+"/domain/"
            with open(data_dir+neg_data_dir+ test1+".txt", "w") as prediction_file:
                new_line = ""
                if test1 == "train":
                    data_now = domain_data_train
                else:
                    data_now = domain_data_test
                for idx, (head, relation, tail, lbl) in enumerate(data_now):
                    new_line += head + "\t" + relation + "\t" + tail + "\t" + str(lbl) + "\n"

                prediction_file.write(new_line)
            with open(data_dir+neg_data_dir+ test1+"_pred.txt", "w") as prediction_file:
                new_line = ""
                if test1 == "train":
                    data_now = domain_data_train1
                else:
                    data_now = domain_data_test1
                for idx, (head, relation, tail, score) in enumerate(data_now):
                    new_line += head + "\t" + relation + "\t" + tail + "\t" + str(score) + "\n"

                prediction_file.write(new_line)

            neg_data_dir = "data/"+test1+"/domainrange/"
            with open(data_dir+neg_data_dir+ test1+".txt", "w") as prediction_file:
                new_line = ""
                if test1 == "train":
                    data_now = domainrange_data_train
                else:
                    data_now = domainrange_data_test
                for idx, (head, relation, tail, lbl) in enumerate(data_now):
                    new_line += head + "\t" + relation + "\t" + tail + "\t" + str(lbl) + "\n"

                prediction_file.write(new_line)
            with open(data_dir+neg_data_dir+ test1+"_pred.txt", "w") as prediction_file:
                new_line = ""
                if test1 == "train":
                    data_now = domainrange_data_train1
                else:
                    data_now = domainrange_data_test1
                for idx, (head, relation, tail, score) in enumerate(data_now):
                    new_line += head + "\t" + relation + "\t" + tail + "\t" + str(score) + "\n"

                prediction_file.write(new_line)

            neg_data_dir = "data/"+test1+"/mix/"
            with open(data_dir+neg_data_dir+ test1+".txt", "w") as prediction_file:
                new_line = ""
                if test1 == "train":
                    data_now = mix_data_train
                else:
                    data_now = mix_data_test
                for idx, (head, relation, tail, lbl) in enumerate(data_now):
                    new_line += head + "\t" + relation + "\t" + tail + "\t" + str(lbl) + "\n"

                prediction_file.write(new_line)
            with open(data_dir+neg_data_dir+ test1+"_pred.txt", "w") as prediction_file:
                new_line = ""
                if test1 == "train":
                    data_now = mix_data_train1
                else:
                    data_now = mix_data_test1
                for idx, (head, relation, tail, score) in enumerate(data_now):
                    new_line += head + "\t" + relation + "\t" + tail + "\t" + str(score) + "\n"

                prediction_file.write(new_line)

            neg_data_dir = "data/"+test1+"/property/"
            with open(data_dir+neg_data_dir+ test1+".txt", "w") as prediction_file:
                new_line = ""
                if test1 == "train":
                    data_now = property_data_train
                else:
                    data_now = property_data_test
                for idx, (head, relation, tail, lbl) in enumerate(data_now):
                    new_line += head + "\t" + relation + "\t" + tail + "\t" + str(lbl) + "\n"

                prediction_file.write(new_line)
            with open(data_dir+neg_data_dir+ test1+"_pred.txt", "w") as prediction_file:
                new_line = ""
                if test1 == "train":
                    data_now = property_data_train1
                else:
                    data_now = property_data_test1
                for idx, (head, relation, tail, score) in enumerate(data_now):
                    new_line += head + "\t" + relation + "\t" + tail + "\t" + str(score) + "\n"

                prediction_file.write(new_line)

            neg_data_dir = "data/"+test1+"/random/"
            with open(data_dir+neg_data_dir+ test1+".txt", "w") as prediction_file:
                new_line = ""
                if test1 == "train":
                    data_now = random_data_train
                else:
                    data_now = random_data_test
                for idx, (head, relation, tail, lbl) in enumerate(data_now):
                    new_line += head + "\t" + relation + "\t" + tail + "\t" + str(lbl) + "\n"

                prediction_file.write(new_line)
            with open(data_dir+neg_data_dir+ test1+"_pred.txt", "w") as prediction_file:
                new_line = ""
                if test1 == "train":
                    data_now = random_data_train1
                else:
                    data_now = random_data_test1
                for idx, (head, relation, tail, score) in enumerate(data_now):
                    new_line += head + "\t" + relation + "\t" + tail + "\t" + str(score) + "\n"

                prediction_file.write(new_line)

            neg_data_dir = "data/"+test1+"/range/"
            with open(data_dir + neg_data_dir + test1+".txt", "w") as prediction_file:
                new_line = ""
                if test1 == "train":
                    data_now = range_data_train
                else:
                    data_now = range_data_test
                for idx, (head, relation, tail, lbl) in enumerate(data_now):
                    new_line += head + "\t" + relation + "\t" + tail + "\t" + str(lbl) + "\n"

                prediction_file.write(new_line)
            with open(data_dir + neg_data_dir + test1+"_pred.txt", "w") as prediction_file:
                new_line = ""
                if test1 == "train":
                    data_now = range_data_train1
                else:
                    data_now = range_data_test1
                for idx, (head, relation, tail, score) in enumerate(data_now):
                    new_line += head + "\t" + relation + "\t" + tail + "\t" + str(score) + "\n"

                prediction_file.write(new_line)






        with open(data_dir+"complete_dataset/"+ "train.txt", "w") as prediction_file:
            new_line = ""
            for idx, (head, relation, tail, lbl) in enumerate(data_train):
                new_line += head + "\t" + relation + "\t" + tail + "\t" + str(lbl) + "\n"

            prediction_file.write(new_line)
        with open(data_dir+"complete_dataset/"+ "train_pred.txt", "w") as prediction_file:
            new_line = ""
            for idx, (head, relation, tail, score) in enumerate(data_train1):
                new_line += head + "\t" + relation + "\t" + tail + "\t" + str(score) + "\n"

            prediction_file.write(new_line)

        with open(data_dir+"complete_dataset/"+"test.txt", "w") as prediction_file:
            new_line = ""
            for idx, (head, relation, tail, lbl) in enumerate(data_test):
                new_line += head + "\t" + relation + "\t" + tail + "\t" + str(lbl) + "\n"

            prediction_file.write(new_line)
        with open(data_dir+"complete_dataset/"+"test_pred.txt", "w") as prediction_file:
            new_line = ""
            for idx, (head, relation, tail, score) in enumerate(data_test1):
                new_line += head + "\t" + relation + "\t" + tail + "\t" + str(score) + "\n"

            prediction_file.write(new_line)



path_dataset_folder = '../dataset/'
se = GenerateTrainTestTriplesSet(data_dir=path_dataset_folder,multiclass=True)
