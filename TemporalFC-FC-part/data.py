from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import pandas as pd
from copy import deepcopy
import string
import torch
import random
random.seed(42)
from urllib.parse import quote, unquote
class Data:
    def __init__(self, args=None):
        complete_dataset  = args.cmp_dataset
        full_hybrid = (args.model == "full-Hybrid")

        sub_dataset_path = "" if (args.sub_dataset_path==None) else args.sub_dataset_path
        prop_split = args.prop_split
        emb_typ = args.emb_type
        bpdp_dataset = (args.eval_dataset == "BPDP")
        emb_folder = str(args.eval_dataset).lower()+"/"
        data_dir = args.path_dataset_folder + emb_folder
        # Quick workaround as we happen to have duplicate triples.
        # None if load complete data, otherwise load parts of dataset with folders in wrong directory.
        # emb_folder = ""
        self.train_set = list((self.load_data(str(data_dir).lower() + "train/"+ sub_dataset_path, data_type="train")))
        self.test_data = list((self.load_data(str(data_dir).lower() + "test/"+ sub_dataset_path, data_type="test")))

        #generate test and validation sets
        self.test_data, self.valid_data = self.generate_test_valid_set(self, self.test_data)

        # self.save_idx_data(str(data_dir).lower() + "train/tkbc_train.txt",self.train_set)

        # self.train_set_pred = list((self.load_data(str(data_dir).lower() + "train/"+ sub_dataset_path, data_type="train_pred", pred=True)))
        # self.test_data_pred = list((self.load_data(str(data_dir).lower() + "test/"+ sub_dataset_path, data_type="test_pred", pred=True)))
        # # factcheck predictions on train and test data
        #
        #
        # self.test_data_pred, self.valid_data_pred = self.generate_test_valid_set(self, self.test_data_pred)

        self.data = self.train_set + list(self.test_data) + list(self.valid_data)
        self.entities = self.get_entities(self.data)

        # self.save_all_resources(self.entities, data_dir, "orignal_data/" , True)


        # self.relations = list(set(self.get_relations(self.train_set) + self.get_relations(self.test_data)))
        self.relations = self.get_relations(self.data)
        if len(self.data[0]) == 5:
            self.times = self.get_times(self.data)
            self.num_times = len(self.times)
            self.idx_times = dict()
            for i in self.times:
                self.idx_times[i] = len(self.idx_times)
        # uncomment it later  when needed
        # if bpdp_dataset:
        #     self.save_all_resources(self.relations, data_dir, "/combined/", False)
        #     # exit(1)
        # elif args.eval_dataset == "FactBench" and prop != None:
        #     self.save_all_resources(self.relations, data_dir, "hybrid_data/combined/properties_split/" + prop.replace("/","_"), False)
        # elif full_hybrid == True:
        #     self.save_all_resources(self.relations, data_dir.replace("hybrid_data/copaal",""), "hybrid_data/combined/" + sub_dataset_path, False)
        # else:
        #     self.save_all_resources(self.relations, data_dir, "hybrid_data/combined/" + sub_dataset_path, False)
        # self.save_all_resources(self.relations, data_dir, "orignal_data/", False)
        self.num_entities = len(self.entities)
        self.num_relations = len(self.relations)
        # self.num_times = len(self.times)


        self.idx_entities = dict()
        self.idx_relations = dict()


        # Generate integer mapping
        for i in self.entities:
            self.idx_entities[i] = len(self.idx_entities)
        for i in self.relations:
            self.idx_relations[i] = len(self.idx_relations)
        neg_data_type = args.negative_triple_generation
        if args.model != "text-only":
            if (os.path.exists('Embeddings/'+emb_typ+'/'+emb_folder + neg_data_type+"-all_entities_embeddings_updated.txt")):
                self.emb_entities = pd.read_csv('Embeddings/'+emb_typ+'/'+emb_folder + neg_data_type+"-all_entities_embeddings_updated.txt", sep="\t")
                self.train_set = self.load_data(data_dir + "train/" + sub_dataset_path, data_type=neg_data_type+"-trainUpdated")
                self.test_data = self.load_data(data_dir + "test/"+ sub_dataset_path, data_type=neg_data_type+"-testUpdated")
                self.valid_data = self.load_data(data_dir + "test/" + sub_dataset_path, data_type=neg_data_type+"-validUpdated")

            else:
                self.emb_entities, self.idx_entities, self.train_set, self.test_data, self.valid_data = \
                    self.get_embeddings(self, self.idx_entities, self.train_set, self.test_data, self.valid_data,
                                        'Embeddings/' + emb_typ + '/' + emb_folder, 'all_entities_embeddings')
                df = pd.DataFrame(self.emb_entities)
                df.to_csv('Embeddings/'+emb_typ+'/'+emb_folder + neg_data_type+"-all_entities_embeddings_updated.txt", index=False, sep="\t")
                self.save_tuples_to_file(self.train_set, data_dir + "train/" + sub_dataset_path + neg_data_type+"-trainUpdated.txt")
                self.save_tuples_to_file(self.test_data, data_dir + "test/" + sub_dataset_path + neg_data_type+ "-testUpdated.txt")
                self.save_tuples_to_file(self.valid_data, data_dir + "test/" + sub_dataset_path + neg_data_type+ "-validUpdated.txt")
        # else:

            self.emb_relation, self.idx_relations, self.train_set, self.test_data, self.valid_data = self.get_embeddings(self,self.idx_relations,  self.train_set, self.test_data, self.valid_data,'Embeddings/'+emb_typ+'/'+emb_folder,'all_relations_embeddings')
            self.emb_times, self.idx_times, self.train_set, self.test_data, self.valid_data = self.get_embeddings(self,self.idx_times, self.train_set, self.test_data, self.valid_data,'Embeddings/'+emb_typ+'/'+emb_folder,'all_times_embeddings')
        # self.emb_entities = self.get_embeddings_pickle(tmp_emb_folder + emb_typ + '/', 'entity.pkl')
        # self.emb_relation = self.get_embeddings_pickle(tmp_emb_folder + emb_typ + '/', 'relation.pkl')
        # self.emb_time = self.get_embeddings_pickle(tmp_emb_folder + emb_typ + '/', 'time.pkl')


            if args.negative_triple_generation == "corrupted-time-based":
                self.train_set = self.generate_negative_triples(self.train_set)
                self.valid_data = self.generate_negative_triples(self.valid_data)
                self.test_data = self.generate_negative_triples(self.test_data)


            print("sentence embeddings parsing started....it may take a while...please wait.....\nFirst time compilation can take aprox 10 minutes")
            if (os.path.exists(data_dir + "train/"+neg_data_type+"trainSEUpdated.csv")):
                self.emb_sentences_train1 = pd.read_csv(data_dir + "train/"+ sub_dataset_path+neg_data_type+"trainSEUpdated.csv", sep="\t")
                self.train_set = self.load_data(data_dir + "train/"+ sub_dataset_path, data_type=neg_data_type+"-trainUpdated")
                # self.emb_sentences_train1 = self.emb_sentences_train1.to_dict()
                self.emb_sentences_train1 = self.dataframe_to_dict(self.emb_sentences_train1)
            else:
                self.emb_sentences_train1, self.train_set = self.get_sen_embeddings(data_dir + "train/" + sub_dataset_path, 'trainSE.csv',
                                                                                     self.train_set)
                # self.save_tuples_to_file(self.emb_sentences_train1,data_dir+"hybrid_data/train/trainSEUpdated.csv")
                df = pd.DataFrame(self.emb_sentences_train1)
                df.to_csv(data_dir + "train/"+ sub_dataset_path+neg_data_type+"trainSEUpdated.csv", index=False, sep="\t")
                self.save_tuples_to_file(self.train_set, data_dir + "train/"+ sub_dataset_path+neg_data_type+"-trainUpdated.txt")

            if (os.path.exists(data_dir + "test/"+neg_data_type+"-testSEUpdated.csv")):
                self.emb_sentences_test1 = pd.read_csv(data_dir + "test/"+ sub_dataset_path+neg_data_type+"-testSEUpdated.csv", sep="\t")
                self.test_data = self.load_data(data_dir + "test/"+ sub_dataset_path, data_type=neg_data_type+"-testUpdated")
                self.emb_sentences_valid1 = pd.read_csv(data_dir + sub_dataset_path+ "test/"+neg_data_type+"-validSEUpdated.csv", sep="\t")
                self.valid_data = self.load_data(data_dir + "test/"+ sub_dataset_path, data_type=neg_data_type+"-validUpdated")
                self.emb_sentences_test1 = self.dataframe_to_dict(self.emb_sentences_test1)
                self.emb_sentences_valid1 = self.dataframe_to_dict(self.emb_sentences_valid1)
                # self.emb_sentences_test1 = self.emb_sentences_test1.to_dict()
                # self.emb_sentences_valid1 = self.emb_sentences_valid1.to_dict()
            else:
                self.emb_sentences_test1, self.emb_sentences_valid1, self.test_data, self.valid_data = self.get_sen_test_valid_embeddings(
                    path=""+data_dir + "test/" + sub_dataset_path, name='testSE.csv', test_data=self.test_data, valid_data=self.valid_data)
                # self.save_tuples_to_file(self.emb_sentences_test1, data_dir + "hybrid_data/test/testSEUpdated.csv")
                self.save_tuples_to_file(self.test_data, data_dir + "test/"+ sub_dataset_path+neg_data_type+"-testUpdated.txt")
                df = pd.DataFrame(self.emb_sentences_test1)
                df.to_csv(data_dir + "test/"+ sub_dataset_path+neg_data_type+"-testSEUpdated.csv", index=False, sep="\t")
                df = pd.DataFrame(self.emb_sentences_valid1)
                df.to_csv(data_dir + "test/"+ sub_dataset_path+neg_data_type+"-validSEUpdated.csv", index=False, sep="\t")
                # self.save_tuples_to_file(self.emb_sentences_valid1, data_dir + "hybrid_data/test/validSEUpdated.csv")
                self.save_tuples_to_file(self.valid_data, data_dir + "test/"+ sub_dataset_path+neg_data_type+"-validUpdated.txt")
        else:
            self.emb_sentences_train1 = pd.read_csv(data_dir + "train/" + sub_dataset_path + neg_data_type + "trainSE.csv", sep="\t")
            self.emb_sentences_test1, self.emb_sentences_valid1, self.test_data, self.valid_data = self.get_sen_test_valid_embeddings(
                path="" + data_dir + "test/" + sub_dataset_path, name=neg_data_type+'testSE.csv', test_data=self.test_data, valid_data=self.valid_data)

        self.emb_sentences_train = self.update_sent_train_embeddings(self, self.emb_sentences_train1)
        self.emb_sentences_test = self.update_sent_train_embeddings(self, self.emb_sentences_test1)
        self.emb_sentences_valid = self.update_sent_train_embeddings(self, self.emb_sentences_valid1)



        self.data = self.train_set + list(self.test_data) + list(self.valid_data)
        self.entities = self.get_entities(self.data)

        self.idx_entities = dict()
        self.idx_relations = dict()


        # Generate integer mapping
        for i in self.entities:
            self.idx_entities[i] = len(self.idx_entities)
        for i in self.relations:
            self.idx_relations[i] = len(self.idx_relations)



        self.idx_train_data = []
        i = 0
        if args.model != "text-only":
            for (s, p, o,year, label) in self.train_set:
                # if args.negative_triple_generation == "corrupted-time-based" and args.model != "temporal" and label==0:
                #     continue
                idx_s, idx_p, idx_o,idx_year, label = self.idx_entities[s], self.idx_relations[p], self.idx_entities[o],self.idx_times[year], label
                self.idx_train_data.append([idx_s, idx_p, idx_o, idx_year, label , i])
                i = i + 1

            self.idx_valid_data = []
            j = 0
            for (s, p, o, year, label) in self.valid_data:
                # if args.negative_triple_generation == "corrupted-time-based" and args.model != "temporal" and label==0:
                #     continue
                idx_s, idx_p, idx_o, idx_year, label = self.idx_entities[s], self.idx_relations[p], self.idx_entities[o],self.idx_times[year], label
                self.idx_valid_data.append([idx_s, idx_p, idx_o, idx_year, label,j])
                j = j + 1

            self.idx_test_data = []
            k = 0
            for (s, p, o, year, label) in self.test_data:
                # if args.negative_triple_generation == "corrupted-time-based" and args.model != "temporal" and label==0:
                #     continue
                idx_s, idx_p, idx_o, idx_year, label = self.idx_entities[s], self.idx_relations[p], self.idx_entities[o], self.idx_times[year], label
                self.idx_test_data.append([idx_s, idx_p, idx_o, idx_year, label,k])
                k = k + 1
        else:
            for (s, p, o, label) in self.train_set:
                # if args.negative_triple_generation == "corrupted-time-based" and args.model != "temporal" and label==0:
                #     continue
                idx_s, idx_p, idx_o, label = self.idx_entities[s], self.idx_relations[p], self.idx_entities[o],  label
                self.idx_train_data.append([idx_s, idx_p, idx_o, label, i])
                i = i + 1

            self.idx_valid_data = []
            j = 0
            for (s, p, o, label) in self.valid_data:
                # if args.negative_triple_generation == "corrupted-time-based" and args.model != "temporal" and label==0:
                #     continue
                idx_s, idx_p, idx_o, label = self.idx_entities[s], self.idx_relations[p], self.idx_entities[o],  label
                self.idx_valid_data.append([idx_s, idx_p, idx_o, label, j])
                j = j + 1

            self.idx_test_data = []
            k = 0
            for (s, p, o, label) in self.test_data:
                # if args.negative_triple_generation == "corrupted-time-based" and args.model != "temporal" and label==0:
                #     continue
                idx_s, idx_p, idx_o, label = self.idx_entities[s], self.idx_relations[p], self.idx_entities[o],  label
                self.idx_test_data.append([idx_s, idx_p, idx_o, label, k])
                k = k + 1

    def generate_negative_triples(self, data):
        data2 = []
        data_final = []
        i =0
        times = []
        for (s, p, o, time, label) in data:
            if label == 1:
                times.append(time)
                data2.append([s, p, o, time, label])
            i = i + 1
        random.shuffle(times)
        data3 = []
        for j in range(len(data2)):
            item = data2.__getitem__(j)
            tim = times.__getitem__(j)
            data3.append([item[0],item[1],item[2],tim,0])

        data_final = data2 + data3
        # data_final.append(data3)
        return data_final
    def save_idx_data(self, path, data):
        print("test")
        with open(path, "w") as f:
            for (s, p, o, year, label) in data:
                f.write("<"+s.replace("<http://dbpedia.org/resource/","").replace(">","") +">\t<"+
                        p.replace("<http://dbpedia.org/ontology/","").replace(">","") +">\t<"+
                        o.replace("<http://dbpedia.org/resource/","").replace(">","") +">\t"+
                        "<occursSince>"+"\t" +
                        year.replace(">","").replace("<","")+"-##-##\n")





    def dataframe_to_dict(self, df):
        data_dict = {}
        for i, row in df.iterrows():
            # Extract the keys (first 3 columns) and values (remaining columns)
            keys = i
            values = row[0:].tolist()
            # Add the key-value pair to the dictionary
            data_dict[keys] = values

        return data_dict.values()
    def save_tuples_to_file(self, tuples, file_name):
        with open(file_name, 'w') as file:
            for t in tuples:
                file.write(str(t[0] +"\t" + t[1] +"\t" +t[2] +"\t"+t[3] +"\t"+ str(True if (t[4]==1) else False)) + '\n')

    def is_valid_test_available(self):
        if len(self.valid_data) > 0 and len(self.test_data) > 0:
            return True
        return False
    @staticmethod
    def save_all_resources(list_all_entities, data_dir, sub_path, entities):
        if entities:
            with open(data_dir+sub_path+'all_entities.txt',"w") as f:
                for item in list_all_entities:
                    f.write("%s\n" % item)
        else:
            with open(data_dir + sub_path + 'all_relations.txt', "w") as f:
                for item in list_all_entities:
                    f.write("%s\n" % item)

    @staticmethod
    def update_and_match_triples_start(self, selected_dataset_data_dir, type, file_name, data_set1, data_set2, properties_split=None, veracity=False):
        if veracity == False:
            if (os.path.exists(selected_dataset_data_dir + type + "/" + file_name)):
                self.set_time_final = list(self.load_data(selected_dataset_data_dir + type + "/", data_type=str(file_name).replace(".txt", ""), pred=True))
            else:
                if len(data_set1) != len(data_set2):
                    self.set_time_final = self.update_match_triples(data_set1, data_set2)
                else:
                    self.set_time_final = data_set2
                self.save_triples(selected_dataset_data_dir, type + "/" + file_name, self.set_time_final)
        else:
            tt = "properties/train/" if (file_name.__contains__("train")) else "properties/test/"
            split = "" if (properties_split == 'None') else tt + "correct/" + properties_split + "_"
            if (os.path.exists(selected_dataset_data_dir + type + "/" + split + file_name)):
                self.set_time_final = list(self.load_data(selected_dataset_data_dir + type + "/" + split, data_type=str(file_name).replace(".txt", ""), pred=True))
            else:
                self.set_time_final = self.update_match_triples(data_set1, data_set2, veracity=veracity)
                self.save_triples(selected_dataset_data_dir, type + "/" + split + file_name, self.set_time_final, veracity=veracity)
        return self.set_time_final

    @staticmethod
    def save_triples(data_dir, type, triples, veracity=False):
        if veracity == False:
            with open(data_dir + type, "w") as f:
                for item in triples:
                    f.write("" + (item[0]) + "\t" + (item[1]) + "\t" + (item[2]) + "\t" + str(item[3]) + "\t" + str(item[4]) + "\n")
        else:
            with open(data_dir + type, "w") as f:
                for item in triples:
                    f.write("" + str(item[0]) + "\t" + str(item[1]) + "\t" + str(item[2]) + "\t" + str(item[3]) + "\n")

    @staticmethod
    def update_match_triples(data_set1, data_set2, veracity=False, final=False):
        data = []
        data_set21 = deepcopy(data_set2)
        # subs = [tp2[0] for tp2 in data_set2]
        # preds = [tp2[1].replace("Of","") for tp2 in data_set2]
        # objs = [tp2[2] for tp2 in data_set2]
        for tp in data_set1:
            found = False
            if veracity == False:
                for tpt in data_set21:
                    # if tpt[0].__contains__('Amadou_Toumani') and (tp[2].__contains__('Amadou_Toumani_')):
                    #     print("test")
                    if (tp[0] == tpt[0] and tp[1] == tpt[1].replace("Of", "") and tp[2] == tpt[2]):
                        data.append([tp[0], tp[1], tp[2], tpt[3], tpt[4]])
                        found = True
                        break
                    elif (tp[2] == tpt[0] and tp[0] == tpt[2]):  # to cover negative triples we are doing like this, swaping the sub and obj and not checking the predicate
                        data.append([tpt[0], tpt[1], tpt[2], tpt[3], 'False'])
                        found = True
                        break

                # if found == False:
                #     print("not found:" + str(tp))
            elif veracity == True and final == True:  # final is for second check
                for tpt in data_set21:
                    if (tp[0] == tpt[0] and tp[1] == tpt[1].replace("Of", "") and tp[2] == tpt[2]):
                        data.append([tp[0], tp[1], tp[2], tp[3], tp[4]])
                        found = True
                        break
                    elif (tp[2] == tpt[0] and tp[1] == tpt[1].replace("Of", "") and tp[0] == tpt[
                        2]):  # to cover negative triples we are doing like this, swaping the sub and obj and not checking the predicate
                        data.append([tp[0], tp[1], tp[2], tp[3], tp[4]])
                        found = True
                        break
                if found == True:
                #     print("not found:" + str(tp))
                # else:
                    data_set21.remove(tpt)

            else:
                for tpt in data_set21:
                    if (tp[0] == tpt[0] and tp[1] == tpt[1].replace("Of", "") and tp[2] == tpt[2]):
                        data.append([tp[0], tp[1], tp[2], tp[3]])
                        found = True
                        break
                    elif (tp[2] == tpt[0] and tp[0] == tpt[2]):  # to cover negative triples we are doing like this, swaping the sub and obj and not checking the predicate
                        data.append([tp[0], tp[1], tp[2], tp[3]])
                        found = True
                        break
                # if found == False:
                #     print("not found:" + str(tp))
                # break

                # else:
                #     print("problematic triple:"+ str(tp))

        return data
    @staticmethod
    def generate_test_valid_set(self, test_data):
        test_set = []
        valid_set = []
        i = 0
        sent_i = 0
        for data in test_data:
            if i % 20 == 0:
                valid_set.append(data)
            else:
                test_set.append(data)

            i += 1
        return  test_set, valid_set

    @staticmethod
    def load_data_with_time(data_dir, data_type, mapped_entities=None, prop=None):
        try:
            data = []
            with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                for datapoint in f:
                    datapoint = datapoint.split("\t")
                    if len(datapoint) >= 5:
                        if len(datapoint) > 5:
                            datapoint[5] = '_'.join(datapoint[4:])
                        s, p, o, time, loc = datapoint[0:5]
                        if prop != None:
                            if not str(p).__eq__(prop + "Of"):
                                continue
                        s = "http://dbpedia.org/resource/" + s
                        if (mapped_entities != None and s in mapped_entities.keys()):
                            s = mapped_entities[s]
                        p = "http://dbpedia.org/ontology/" + p
                        o = "http://dbpedia.org/resource/" + o
                        if (mapped_entities != None and o in mapped_entities.keys()):
                            o = mapped_entities[o]
                        data.append(("<" + s + ">", "<" + p + ">", "<" + o + ">", time, "True"))
                    elif len(datapoint) == 3:
                        s, p, label = datapoint
                        assert label == 'True' or label == 'False'
                        if label == 'True':
                            label = 1
                        else:
                            label = 0
                        data.append((s, p, 'DUMMY', label))
                    else:
                        raise ValueError
        except FileNotFoundError as e:
            print(e)
            print('Add empty.')
            data = []
        return data
    @staticmethod
    def load_data(data_dir, data_type, pred=False):
        try:
            data = []
            if pred == False:
                with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                    for datapoint in f:
                        datapoint = datapoint.split()
                        if len(datapoint) == 4:
                            s, p, o, label = datapoint
                            if label == 'True':
                                label = 1
                            else:
                                label = 0
                            data.append((s, p, o, label))
                        elif len(datapoint) == 3:
                            s, p, label = datapoint
                            assert label == 'True' or label == 'False'
                            if label == 'True':
                                label = 1
                            else:
                                label = 0
                            data.append((s, p, 'DUMMY', label))
                        elif len(datapoint) == 5:
                            s, p, o,year, label = datapoint
                            if label == 'True':
                                label = 1
                            else:
                                label = 0
                            year = year.replace(".0","").replace('<','').replace('>','')
                            if int(year)>=1900 and int(year)<=2022:
                                data.append((s, p, o,"<"+year.replace(".0","")+">", label))
                        else:
                            raise ValueError
            else:
                with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                    for datapoint in f:
                        datapoint = datapoint.split()
                        if len(datapoint) == 5:
                            s, p, o, label, dot = datapoint
                            data.append((s, p, o, label.replace("\"^^<http://www.w3.org/2001/XMLSchema#double>","")))
                        elif len(datapoint) == 4:
                            s, p, o, label = datapoint
                            data.append((s, p, o, label))
                        elif len(datapoint) == 3:
                            s, p, label = datapoint
                            data.append((s, p, 'DUMMY', label))
                        else:
                            raise ValueError
        except FileNotFoundError as e:
            print(e)
            print('Add empty.')
            data = []
        return data

    @staticmethod
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations
    def get_times(self,data):
        times = sorted(list(set([d[3] for d in data])))
        return times
    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities
    # update the embeddings manually with tabs instead of commas if commas are there
    # / home / umair / Documents / pythonProjects / HybridFactChecking / Embeddings / ConEx_dbpedia
    def get_embeddings_pickle(self,path,name):
        # embeddings = dict()
        embd = torch.load("%s%s" % (path,name),map_location=torch.device('cpu'))
        return embd.weight
    @staticmethod
    def get_embeddings(self, idxs, train_set, test_data,valid_data, path,name):
        missing_entities = []
        embeddings = dict()
        # print("%s%s.txt" % (path,name))
        idx2 = dict()
        for tt in idxs.keys():
            idx2[self.remove_punctuation(tt)] = idxs.get(tt)
        if not os.path.exists("%s%s.txt" % (path,name)):
            print("Can't find embeddings in the given path...please change!")
        with open("%s%s.txt" % (path,name), "r") as f:
            for datapoint in f:
                if datapoint.startswith("0\t"):
                    continue
                elif datapoint.startswith("0,1,2,3"):
                    print("embeddings files should be tab seperated.")
                    exit(1)

                data = datapoint.split('>')
                if len(data)==1:
                    print("stoped: getting embeddings function")
                    exit(1)

                if len(data) > 1:
                    data2 = data[0]+">",data[1].split('\t')[1:]
                    # test = data2[0].replace("\"","").replace("_com",".com").replace("Will-i-am","Will.i.am").replace("Will_i_am","Will.i.am")
                    # test = data2[0].replace("\"","")
                    if self.remove_punctuation(data2[0]) in idx2:
                    # if data2[0] in idx2:
                        embeddings[data2[0]] = data2[1]
        for emb in idxs:
            if emb not in embeddings.keys():
                missing_entities.append(emb)
                # print("this is missing in embeddings file:"+ emb)
                # exit(1)
        subjects = [t[0] for t in train_set]
        objects = [t[0] for t in train_set]
        subjects1 = [t[0] for t in test_data]
        objects1 = [t[0] for t in test_data]
        subjects3 = [t[0] for t in valid_data]
        objects3 = [t[0] for t in valid_data]
        if len(missing_entities) > 0:
            for ent in missing_entities:
                idxs.__delitem__(ent)

                if (ent in subjects) or (ent in objects):
                    for tt in train_set:
                        if tt[0] == ent or tt[2] == ent:
                            train_set.remove(tt)
                if (ent in subjects1) or (ent in objects1):
                    for tt in test_data:
                        if tt[0] == ent or tt[2] == ent:
                            test_data.remove(tt)
                if (ent in subjects3) or (ent in objects3):
                    for tt in valid_data:
                        if tt[0] == ent or tt[2] == ent:
                            valid_data.remove(tt)

        if len(idxs) > len(embeddings):
            print("embeddings missing")
            exit(1)
        embeddings_final = dict()
        for emb in idxs.keys():
            if emb in embeddings.keys():
                embeddings_final[emb] = embeddings[emb]
            else:
                print('no embedding', emb)
                exit(1)

        return embeddings_final.values(), idxs, train_set, test_data, valid_data

    @staticmethod
    def get_comma_seperated_embeddings(idxs, path, name):
        embeddings = dict()
        # print("%s%s.txt" % (path,name))
        with open("%s%s.txt" % (path, name), "r") as f:
            for datapoint in f:
                data = datapoint.split('> ,')
                if datapoint.startswith("<http://dbpedia.org/resource/Abu_Jihad_("):
                    print(datapoint)
                if len(data) == 1:
                    data = datapoint.split('>\",')
                if len(data) > 1:
                    data2 = data[0] + ">", data[1].split(',')
                    # test = data2[0].replace("\"","").replace("_com",".com").replace("Will-i-am","Will.i.am").replace("Will_i_am","Will.i.am")
                    test = data2[0].replace("\"", "")
                    if test in idxs:
                        embeddings[test] = data2[1]
                    # else:
                    #     print('Not in embeddings:',datapoint)
                    # exit(1)
                # else:
                #     print('Not in embeddings:',datapoint)
                #     exit(1)
        for emb in idxs:
            if emb not in embeddings.keys():
                print("this is missing in embeddings file:" + emb)
                exit(1)

        if len(idxs) > len(embeddings):
            print("embeddings missing")
            exit(1)
        embeddings_final = dict()
        for emb in idxs.keys():
            if emb in embeddings.keys():
                embeddings_final[emb] = embeddings[emb]
            else:
                print('no embedding', emb)
                exit(1)

        return embeddings_final.values()
    @staticmethod
    def get_copaal_veracity(path, name, train_data):
        emb = dict()

        embeddings_train = dict()
        # print("%s%s" % (path,name))

        i = 0
        train_i = 0
        found = False
        with open("%s%s" % (path, name), "r") as f:
            for datapoint in f:
                if datapoint.startswith("0\t1\t2"):
                    continue
                else:
                    emb[i] = datapoint.split('\t')
                    try:
                        for dd in train_data:
                            # figure out some way to handle this first argument well
                            if (emb[i][0] == dd[0].replace(',', '')) and (emb[i][1] == dd[1].replace(',', '')) and (
                                    emb[i][2] == dd[2].replace(',', '')):
                                # print('train data found')
                                embeddings_train[train_i] =np.append(emb[i][:3],emb[i][-1].replace("\n",""))
                                train_i += 1
                                found = True
                                break

                            # else:
                            #     print('error')
                            # exit(1)
                    except:
                        print('ecception')
                        exit(1)
                    if found == False:
                        if (train_i >= len(train_data)):
                            break
                        else:
                            print("some training data missing....not found:" + str(emb[i]))
                            exit(1)
                    i = i + 1
                    found = False

                    # i = i+1
            embeddings_train_final = dict()
            jj = 0
            # print("sorting")
            for embb in train_data:
                ff = False
                for embb2 in embeddings_train.values():
                    if ((embb[0].replace(',', '') == embb2[0].replace(',', '')) and (
                            embb[1].replace(',', '') == embb2[1].replace(',', '')) and (
                            embb[2].replace(',', '') == embb2[2].replace(',', ''))):
                        embeddings_train_final[jj] = embb2
                        jj = jj + 1
                        ff = True
                        break
                if ff == False:
                    print("problem: not found")
                    exit(1)

        if len(train_data) != len(embeddings_train_final):
            print("problem")
            exit(1)
        return embeddings_train_final.values()
    @staticmethod
    def update_entity(self, ent):
        ent = ent.replace("+", "")
        if (ent.__contains__("&") or ent.__contains__("%")) and (
                (not ent.__contains__("%3F")) and (not ent.__contains__("%22"))):
            sub2 = ""

            for chr in ent:
                if chr == "&" or chr == "%":
                    break
                else:
                    sub2 += chr
            if ent[0]=="<":
                ent = sub2 + ">"
            else:
                ent = sub2

        if ent.__contains__("?"):
            ent = ent.replace("?", "%3F")

        if ent.__contains__("\"\""):
            ent= ent.replace("\"\"", "%22")
        if ent[0] == "\"" and ent[-1] == "\"":
            ent = ent[1:-1]
        if ent[0] == "\'" and ent[-1] == "\'":
            ent = ent[1:-1]
        if ent[0] == "<" and ent[-1] == ">":
            ent = ent[1:-1]
        return ent
    def remove_punctuation(self,  str1):
        punctuations = string.punctuation
        if str1.__contains__("&"):
            str1=str1.split('&')[0]
        str7 = "".join(char for char in str1 if char not in punctuations)
        return str7

    def without(self,d, key):
        new_d = d.copy()
        new_d2 = dict()
        new_d.pop(key)
        count = 0
        for dd in new_d.values():
            new_d2[count]=dd
            count+=1
        return new_d2


    def get_sen_embeddings(self, path, name, train_data):
        # Load sentence embeddings from file
        embeddings = {}
        embeddings_triple = {}
        embeddings_only = {}
        embeddings_subject = {}
        embeddings_predicate = {}
        embeddings_object = {}
        embeddings_final = []
        idx = 0
        with open(os.path.join(path, name), "r") as f:
            for line in f:
                if line.startswith("0\t1\t2"):
                    continue
                fields = line.split('\t')
                embeddings[idx] = [(field) for field in fields[0:]]
                embeddings[idx][:3] = [self.remove_punctuation(field) for field in fields[:3]]
                embeddings_triple[idx] = [self.remove_punctuation(field) for field in fields[:3]]
                embeddings_only[idx] = [field for field in fields[3:]]
                embeddings_subject[idx] = self.remove_punctuation(fields[0])
                embeddings_predicate[idx] = self.remove_punctuation(fields[1])
                embeddings_object[idx] = self.remove_punctuation(fields[2])
                idx = idx +1
                # if idx>400:
                #     break
        single_list = [item for sublist in list(embeddings_triple.values()) for item in sublist]
        # Filter train data based on presence in sentence embeddings
        filtered_train_data = []
        len_triples =  len(embeddings_subject)
        zipped_file = zip(embeddings_subject.values(), embeddings_predicate.values(),embeddings_object.values(), embeddings_only.values())
        kk = 0

        for data in train_data:
            found = False
            if kk >= len_triples:
                if all(self.remove_punctuation(field) in single_list for field in data[:3]):
                    zipped_file = zip(embeddings_subject.values(), embeddings_predicate.values(),embeddings_object.values(), embeddings_only.values())
                    kk=0
            for i, (s,p,o,e) in enumerate(zipped_file):
                kk = kk+1
                if s==self.remove_punctuation(data[0]) and p == self.remove_punctuation(data[1]) and o == self.remove_punctuation(data[2]):
                    filtered_train_data.append(data)
                    embeddings_final.append(data[:3]+e)
                    found = True
                    break

            if found == False:
                if kk >= len_triples:
                    if all(self.remove_punctuation(field) in single_list for field in data[:3]):
                        zipped_file = zip(embeddings_subject.values(), embeddings_predicate.values(),embeddings_object.values(), embeddings_only.values())
                        kk = 0
                        for i, (s, p, o,e) in enumerate(zipped_file):
                            kk = kk + 1
                            if s == self.remove_punctuation(data[0]) and p == self.remove_punctuation(data[1]) and o == self.remove_punctuation(data[2]):
                                filtered_train_data.append(data)
                                embeddings_final.append(data[:3] + e)
                                found = True
                                break

        # Extract embeddings for filtered train data
        zipped_file2 = zip(embeddings_final)
        embeddings_train = []
        for data in filtered_train_data:
            for i, (emb) in enumerate(zipped_file2):
                # if all(self.remove_punctuation(field) in emb[:3] for field in data[:3]):
                emb = emb[0]
                emb[-1] = emb[-1].replace("'", "").replace("\n", "")
                if len(emb) == (768 * 3) + 3 + 1:
                    embeddings_train.append(data[:3]+emb[3:-1])
                elif len(emb) == (768 * 3) + 3:
                    embeddings_train.append(data[:3]+emb[:3])
                else:
                    raise ValueError(f"There is something fishy: {emb}")
                break

        return embeddings_train, filtered_train_data

    @staticmethod
    def update_copaal_veracity_score(self, train_emb):
        embeddings_train = dict()
        i = 0
        for train in train_emb:
            embeddings_train[i] = train[3:]
            i += 1

        return embeddings_train.values()

    @staticmethod
    def update_veracity_train_data(self, train_emb):
        embeddings_train = dict()
        i = 0
        for train in train_emb:
            embeddings_train[i] = train[3:]
            i += 1

        return embeddings_train.values()
    @staticmethod
    def update_sent_train_embeddings(self, train_emb):
        embeddings_train = dict()
        i=0
        if isinstance(train_emb, pd.core.frame.DataFrame):
            new_df =pd.DataFrame(train_emb).iloc[:,3:]
            for idx, train in new_df.iterrows():
                embeddings_train[i] = train.values
                i+=1
        else:
            for train in train_emb:
                embeddings_train[i] = train[3:]
                i += 1

        return embeddings_train.values()

    @staticmethod
    def get_veracity_test_valid_data(path, name, test_data, valid_data):
        embeddings_test, embeddings_valid = dict(), dict()
        emb = dict()
        # print("%s%s" % (path, name))
        found = False
        i = 0
        test_i = 0
        valid_i = 0
        with open("%s%s" % (path, name), "r") as f:
            for datapoint in f:
                if datapoint.startswith("0\t1\t2"):
                    continue
                else:
                    emb[i] = datapoint.split('\t')
                    try:
                        for dd in test_data:
                            # figure out some way to handle this first argument well
                            if (emb[i][0].replace(',', '') == dd[0].replace(',', '')) and (
                                    emb[i][1].replace(',', '') == dd[1].replace(',', '')) and (
                                    emb[i][2].replace(',', '') == dd[2].replace(',', '')):
                                # print('test data found')
                                embeddings_test[test_i] = np.append(emb[i][:3],emb[i][-1].replace("\n",""))
                                test_i += 1
                                found = True
                                break
                        for vd in valid_data:
                            # figure out some way to handle this first argument well
                            if (emb[i][0].replace(',', '') == vd[0].replace(',', '')) and (
                                    emb[i][1].replace(',', '') == vd[1].replace(',', '')) and (
                                    emb[i][2].replace(',', '') == vd[2].replace(',', '')):
                                # print('valid data found')
                                embeddings_valid[valid_i] = np.append(emb[i][:3],emb[i][-1].replace("\n",""))
                                valid_i += 1
                                found = True
                                break
                        if found == False:
                            print("some data missing from test and validation sets..error" + str(emb[i]))
                            exit(1)
                        else:
                            found = False

                    except:
                        print('ecception')
                        exit(1)
                    i = i + 1

        embeddings_test_final, embeddings_valid_final = dict(), dict()
        i = 0
        for dd in test_data:
            for et in embeddings_test.values():
                if (et[0].replace(',', '') == dd[0].replace(',', '')) and (
                        et[1].replace(',', '') == dd[1].replace(',', '')) and (
                        et[2].replace(',', '') == dd[2].replace(',', '')):
                    embeddings_test_final[i] = et
                    i = i + 1
                    break
        i = 0
        for dd in valid_data:
            # print(dd)
            for et in embeddings_valid.values():
                if (et[0].replace(',', '') == dd[0].replace(',', '')) and (
                        et[1].replace(',', '') == dd[1].replace(',', '')) and (
                        et[2].replace(',', '') == dd[2].replace(',', '')):
                    embeddings_valid_final[i] = et
                    i = i + 1
                    break
        if (len(embeddings_valid_final) != len(valid_data)) and (len(embeddings_test_final) != len(test_data)):
            exit(1)
        return embeddings_test_final.values(), embeddings_valid_final.values()

    def get_sen_test_valid_embeddings(self, path, name, test_data, valid_data):
        # Load sentence embeddings from file
        embeddings = {}
        embeddings_triple = {}
        embeddings_only = {}
        embeddings_subject = {}
        embeddings_predicate = {}
        embeddings_object = {}
        test_embeddings_final = []
        valid_embeddings_final = []
        idx = 0
        with open(os.path.join(path, name), "r") as f:
            for line in f:
                if line.startswith("0\t1\t2"):
                    continue
                fields = line.split('\t')
                embeddings[idx] = [(field) for field in fields[0:]]
                embeddings[idx][:3] = [self.remove_punctuation(field) for field in fields[:3]]
                embeddings_triple[idx] = [self.remove_punctuation(field) for field in fields[:3]]

                embeddings_only[idx] = [field for field in fields[3:]]
                embeddings_subject[idx] = self.remove_punctuation(fields[0])
                embeddings_predicate[idx] = self.remove_punctuation(fields[1])
                embeddings_object[idx] = self.remove_punctuation(fields[2])

                idx = idx +1
        single_list = [item for sublist in list(embeddings_triple.values()) for item in sublist]
        # Filter train data based on presence in sentence embeddings
        filtered_test_data = []
        len_triples = len(embeddings_subject)
        zipped_file = zip(embeddings_subject.values(), embeddings_predicate.values(), embeddings_object.values(), embeddings_only.values())
        kk = 0

        for data in test_data:
            found = False
            if kk >= len_triples:
                if all(self.remove_punctuation(field) in single_list for field in data[:3]):
                    zipped_file = zip(embeddings_subject.values(), embeddings_predicate.values(), embeddings_object.values(), embeddings_only.values())
                    kk = 0
            for i, (s, p, o, e) in enumerate(zipped_file):
                kk = kk + 1
                if s == self.remove_punctuation(data[0]) and p == self.remove_punctuation(data[1]) and o == self.remove_punctuation(data[2]):
                    filtered_test_data.append(data)
                    test_embeddings_final.append(list(data[:3]) + e)
                    found = True
                    break

            if found == False:
                if kk >= len_triples:
                    if all(self.remove_punctuation(field) in single_list for field in data[:3]):
                        zipped_file = zip(embeddings_subject.values(), embeddings_predicate.values(), embeddings_object.values(), embeddings_only.values())
                        kk = 0
                        for i, (s, p, o, e) in enumerate(zipped_file):
                            kk = kk + 1
                            if s == self.remove_punctuation(data[0]) and p == self.remove_punctuation(data[1]) and o == self.remove_punctuation(data[2]):
                                filtered_test_data.append(data)
                                test_embeddings_final.append(data[:3] + e)
                                found = True
                                break


        filtered_valid_data = []

        for data in valid_data:
            found = False
            if kk >= len_triples:
                if all(self.remove_punctuation(field) in single_list for field in data[:3]):
                    zipped_file = zip(embeddings_subject.values(), embeddings_predicate.values(), embeddings_object.values(), embeddings_only.values())
                    kk = 0
            for i, (s, p, o, e) in enumerate(zipped_file):
                kk = kk + 1
                if s == self.remove_punctuation(data[0]) and p == self.remove_punctuation(data[1]) and o == self.remove_punctuation(data[2]):
                    filtered_valid_data.append(data)
                    valid_embeddings_final.append(list(data[:3]) + e)
                    found = True
                    break

            if found == False:
                if kk >= len_triples:
                    if all(self.remove_punctuation(field) in single_list for field in data[:3]):
                        zipped_file = zip(embeddings_subject.values(), embeddings_predicate.values(), embeddings_object.values(), embeddings_only.values())
                        kk = 0
                        for i, (s, p, o, e) in enumerate(zipped_file):
                            kk = kk + 1
                            if s == self.remove_punctuation(data[0]) and p == self.remove_punctuation(data[1]) and o == self.remove_punctuation(data[2]):
                                filtered_valid_data.append(data)
                                valid_embeddings_final.append(data[:3] + e)
                                found = True
                                break



        # Extract embeddings for filtered train data
        zipped_file2 = zip(test_embeddings_final)
        embeddings_test = []
        for data in filtered_test_data:
            for i, (emb) in enumerate(zipped_file2):
                # if all(self.remove_punctuation(field) in emb[:3] for field in data[:3]):
                emb = emb[0]
                emb[-1] = emb[-1].replace("'", "").replace("\n", "")
                if len(emb) == (768 * 3) + 3 + 1:
                    embeddings_test.append(list(data[:3]) + emb[3:-1])
                elif len(emb) == (768 * 3) + 3:
                    embeddings_test.append(list(data[:3]) + emb[3:-1])
                else:
                    raise ValueError(f"There is something fishy: {emb}")
                break

        zipped_file2 = zip(valid_embeddings_final)
        embeddings_valid = []
        for data in filtered_valid_data:
            for i, (emb) in enumerate(zipped_file2):
                # if all(self.remove_punctuation(field) in emb[:3] for field in data[:3]):
                emb = emb[0]
                emb[-1] = emb[-1].replace("'", "").replace("\n", "")
                if len(emb) == (768 * 3) + 3 + 1:
                    embeddings_valid.append(list(data[:3]) + emb[3:-1])
                elif len(emb) == (768 * 3) + 3:
                    embeddings_valid.append(list(data[:3]) + emb[3:-1])
                else:
                    raise ValueError(f"There is something fishy: {emb}")
                break

        return embeddings_test, embeddings_valid, filtered_test_data, filtered_valid_data
    @staticmethod
    def get_sent_test_valid_embeddings(self, path, name, test_data, valid_data):
        embeddings_test, embeddings_valid = dict(),dict()
        emb = dict()
        # print("%s%s" % (path, name))
        found = False
        i = 0
        test_i = 0
        valid_i = 0
        test_data1 = np.array(test_data)
        valid_data1 = np.array(valid_data)
        with open("%s%s" % (path, name), "r") as f:
            for datapoint in f:
                if datapoint.startswith("0\t1\t2"):
                    continue
                else:
                    emb[i] = datapoint.split('\t')
                    try:
                        if emb[i][0] != "\"0\"":
                            str1 = self.remove_punctuation(emb[i][0])
                            str2 = self.remove_punctuation(emb[i][1])
                            str3 = self.remove_punctuation(emb[i][2])
                            str4 = self.remove_punctuation(test_data1[test_i][0])
                            str5 = self.remove_punctuation(test_data1[test_i][1])
                            str6 = self.remove_punctuation(test_data1[test_i][2])
                            if ((str1).__contains__(str4) and
                                    (str2).__contains__(str5) and
                                    (str3).__contains__(str6)):

                                emb[i][-1] = emb[i][-1].replace("'", "").replace("\n", "")
                                if (len(emb[i])) == ((768 * 3) + 3 + 1):
                                    # because defacto scores are also appended at the end
                                    embeddings_test[test_i] = emb[i][:-1]
                                elif (len(emb[i])) == ((768 * 3) + 3):
                                    # emb[i][-1] = emb[i]
                                    embeddings_test[test_i] = emb[i]
                                else:
                                    print("there is something fishy:" + str(emb[i]))
                                    exit(1)
                                # embeddings_test[test_i] = emb[i]
                                test_i += 1
                                found = True
                            else:
                                found = False


                                # if (test_i >= len(test_data)):  # len(train_data)
                                #     break



                            if found == False:
                                str4 = self.remove_punctuation(valid_data1[valid_i][0])
                                str5 = self.remove_punctuation(valid_data1[valid_i][1])
                                str6 = self.remove_punctuation(valid_data1[valid_i][2])
                                if ((str1).__contains__(str4) and
                                        (str2).__contains__(str5) and
                                        (str3).__contains__(str6)):

                                    emb[i][-1] = emb[i][-1].replace("'", "").replace("\n", "")
                                    if (len(emb[i])) == ((768 * 3) + 3 + 1):
                                        # because defacto scores are also appended at the end
                                        embeddings_valid[valid_i] = emb[i][:-1]
                                    elif (len(emb[i])) == ((768 * 3) + 3):
                                        # emb[i][-1] = emb[i]
                                        embeddings_valid[valid_i] = emb[i]
                                    else:
                                        print("there is something fishy:" + str(emb[i]))
                                        exit(1)
                                    # embeddings_test[valid_i] = emb[i]
                                    valid_i += 1
                                    found = True
                                else:
                                    found = False
                                # rechecking the previous data......exponential
                                if found == False:
                                    jj = 0
                                    for t_data in test_data1:
                                        str4 = self.remove_punctuation(t_data[0])
                                        str5 = self.remove_punctuation(t_data[1])
                                        str6 = self.remove_punctuation(t_data[2])
                                        if ((str1).__contains__(str4) and
                                                (str2).__contains__(str5) and
                                                (str3).__contains__(str6)):
                                            emb[i][-1] = emb[i][-1].replace("'", "").replace("\n", "")
                                            if (len(emb[i])) == ((768 * 3) + 3 + 1):
                                                # because defacto scores are also appended at the end
                                                embeddings_test[test_i] = emb[i][:-1]
                                            elif (len(emb[i])) == ((768 * 3) + 3):
                                                # emb[i][-1] = emb[i]
                                                embeddings_test[test_i] = emb[i]
                                                print("success in previous data search")
                                            else:
                                                print("there is something fishy:" + str(emb[i]))
                                                exit(1)
                                            j = jj
                                            test_i += 1
                                            found = True
                                            break
                                        else:
                                            found = False
                                        jj = jj + 1
                                # if found == False:
                                #     print(emb[i])

                                if found == False:
                                    jj = 0
                                    for t_data in valid_data1:
                                        str4 = self.remove_punctuation(t_data[0])
                                        str5 = self.remove_punctuation(t_data[1])
                                        str6 = self.remove_punctuation(t_data[2])
                                        if ((str1).__contains__(str4) and
                                                (str2).__contains__(str5) and
                                                (str3).__contains__(str6)):
                                            emb[i][-1] = emb[i][-1].replace("'", "").replace("\n", "")
                                            if (len(emb[i])) == ((768 * 3) + 3 + 1):
                                                # because defacto scores are also appended at the end
                                                embeddings_valid[valid_i] = emb[i][:-1]
                                            elif (len(emb[i])) == ((768 * 3) + 3):
                                                # emb[i][-1] = emb[i]
                                                embeddings_valid[valid_i] = emb[i]
                                                print("success in previous data search")
                                            else:
                                                print("there is something fishy:" + str(emb[i]))
                                                exit(1)
                                            j = jj
                                            valid_i += 1
                                            found = True
                                            break
                                        else:
                                            found = False
                                        jj = jj + 1
                            # if found == False:
                            #     print("Not found: "+datapoint)
                            # else:
                            #     print("found")
                            # print("test data"+str(test_i))
                            # print("valid data"+str(valid_i))
                            # if test_i >= 100:
                            #     break
                        found = False
                    except Exception as e:
                        print(emb[i])
                        print('exception--'+str(e))
                        # exit(1)
                    i = i + 1

        if (len(embeddings_valid)!= len(valid_data)) and (len(embeddings_test)!= len(test_data)):
            print("check lengths of valid and test data:valid_emb:"+str(len(embeddings_valid))+
                  " valid_data"+str(len(valid_data))+
                  "test_data:"+str(len(test_data))+"test_emb:"+str(len(embeddings_test)))
            # exit(1)
        train_i = 0
        test_data_copy = deepcopy(test_data)
        valid_data_copy = deepcopy(valid_data)
        embeddings_test_final = dict()
        embeddings_valid_final = dict()
        for dd in test_data:
            found_data = False
            jj = 0
            for sd in embeddings_test.values():
                sub = self.remove_punctuation( dd[0])
                pred = self.remove_punctuation( dd[1])
                obj = self.remove_punctuation( dd[2])
                sub1 =  self.remove_punctuation( sd[0])
                pred1 = self.remove_punctuation( sd[1])
                obj1 =  self.remove_punctuation( sd[2])
                if (((sub == sub1) and (pred == pred1) and (obj == obj1))
                        or
                        ((sub1.lower() == sub.lower()) and (pred1.lower() == pred.lower()) and
                         (obj1.lower() == obj.lower()))):
                    embeddings_test_final[train_i] = sd
                    train_i += 1
                    found_data = True
                    break
                jj += 1
            if found_data == False:
                test_data_copy.remove(dd)
                print("missing test data from sentence embeddings file:" + str(dd))
            else:
                # embeddings_test.pop(jj)
                embeddings_test = self.without(embeddings_test, jj)
                # print("to delete from list: " + str(sd))
                # del embeddings_test[sd]

        test_data = deepcopy(test_data_copy)

        train_i = 0
        for dd in valid_data:
            found_data = False
            jj = 0
            for sd in embeddings_valid.values():
                sub = self.remove_punctuation( dd[0])
                pred = self.remove_punctuation( dd[1])
                obj = self.remove_punctuation( dd[2])
                sub1 =  self.remove_punctuation( sd[0])
                pred1 = self.remove_punctuation( sd[1])
                obj1 =  self.remove_punctuation( sd[2])
                if (((sub == sub1) and (pred == pred1) and (obj == obj1))
                        or
                    ((sub1.lower() == sub.lower()) and (pred1.lower() == pred.lower()) and (obj1.lower() == obj.lower()))):
                    embeddings_valid_final[train_i] = sd
                    train_i += 1
                    found_data = True
                    break
                jj += 1
            if found_data == False:
                valid_data_copy.remove(dd)
                print("missing valid data from sentence embeddings file:" + str(dd))
            else:
                # print("to delete from list: " + str(sd))
                # embeddings_valid.pop(jj)
                embeddings_valid = self.without(embeddings_valid, jj)


        valid_data = deepcopy(valid_data_copy)

        return embeddings_test_final.values(), embeddings_valid_final.values(), test_data, valid_data


        # return embeddings.values()

    @staticmethod
    def get_sent_embeddings(self, path, name, train_data):
        emb = dict()
        embeddings_train = dict()
        # print("%s%s" % (path,name))
        train_data_copy = deepcopy(train_data)
        train_data1 = np.array(train_data_copy)
        i = 0
        j = 0
        k = 0
        jj = 0
        train_i = 0
        found = False
        sentence_emb = []
        with open("%s%s" % (path, name), "r") as f:
            for datapoint in f:
                if datapoint.startswith("0\t1\t2"):
                    continue
                else:
                    if datapoint.startswith("http://dbpedia.org/resource/Vlado_Brankovic"):
                        print("test")
                    emb[i] = datapoint.split('\t')
                    # sentence_emb.append(emb[])
                    i = i + 1

        print("read")
        subjects = []
        predicates = []
        objects = []
        subjects1 = []
        predicates1 = []
        objects1 = []
        train_data_copy = deepcopy(train_data)
        for data in emb.values():
            subjects.append(self.remove_punctuation(data[0]))
            predicates.append(self.remove_punctuation(data[1]))
            objects.append(self.remove_punctuation(data[2]))
        for data in train_data1:
            subjects1.append(self.remove_punctuation(data[0]))
            predicates1.append(self.remove_punctuation(data[1]))
            objects1.append(self.remove_punctuation(data[2]))

        for trainee in train_data1:
            str4 = self.remove_punctuation(trainee[0])
            str5 = self.remove_punctuation(trainee[1])
            str6 = self.remove_punctuation(trainee[2])
            if (str4 not in subjects) or (str5 not in predicates) or (str6 not in objects):
                # train_data_copy.remove(trainee)
                continue
            for (key1, emb1) in zip(emb.keys(), emb.values()):
                # remove punctuation from the string
                str1 = self.remove_punctuation(emb1[0])
                str2 = self.remove_punctuation(emb1[1])
                str3 = self.remove_punctuation(emb1[2])
                if (str1 not in subjects1) or (str2 not in predicates1) or (str3 not in objects1):
                    # emb.pop(key1)
                    continue

                try:
                    if emb1 != "0":

                        if ((str1).__contains__(str4) and
                                (str2).__contains__(str5) and
                                (str3).__contains__(str6)):
                            emb1[-1] = emb1[-1].replace("'", "").replace("\n", "")
                            if (len(emb1)) == ((768 * 3) + 3 + 1):
                                # because defacto scores are also appended at the end
                                embeddings_train[train_i] = emb1[:-1]
                            elif (len(emb1)) == ((768 * 3) + 3):
                                # emb[i][-1] = emb[i]
                                embeddings_train[train_i] = emb1
                            else:
                                print("there is something fishy:" + str(emb1))
                                exit(1)
                            # train_data1.__delitem__(j)
                            train_i += 1
                            found = True
                        else:
                            found = False
                except:
                    print('ecception')
                    os.remove("%s%s" % (path, self.neg_data_type + "-trainSEUpdated.csv"))
                    os.remove("%s%s" % (path, self.neg_data_type + "-trainUpdated.csv"))
                    exit(1)
                # if (train_i >= 8000):  # len(train_data)
                #     break
                if found == False:
                    jj = 0
                else:
                    break
                found = False

                # i = i+1

        if len(train_data) != len(embeddings_train):
            print("problem: length of train and sentence embeddings arrays are different:train:" + str(len(train_data)) + ",emb:" + str(len(embeddings_train)))
            # exit(1)
        # following code is just for ordering the data in sentence vectors
        train_i = 0
        train_data = train_data_copy
        embeddings_train_final = dict()
        try:
            for dd in train_data:
                found_data = False
                jj = 0
                for sd in embeddings_train.values():
                    sub = self.remove_punctuation(dd[0])  # to be updated please
                    pred = self.remove_punctuation(dd[1])
                    obj = self.remove_punctuation(dd[2])
                    sub1 = self.remove_punctuation(sd[0])
                    pred1 = self.remove_punctuation(sd[1])
                    obj1 = self.remove_punctuation(sd[2])

                    if (((sub == sub1) and (pred == pred1) and (obj == obj1))
                            or
                            ((sub1.lower() == sub.lower()) and (pred1.lower() == pred.lower()) and
                             (obj1.lower() == obj.lower()))):
                        embeddings_train_final[train_i] = sd
                        train_i += 1
                        found_data = True
                        break
                    jj += 1
                if found_data == False:
                    train_data_copy.remove(dd)
                    print("missing train data from sentence embeddings file:" + str(dd))
                else:
                    # print("to delete from list: "+str(sd))
                    embeddings_train = self.without(embeddings_train, jj)
                    # embeddings_train = embeddings_train.dropna().reset_index(drop=True)
                    # del embeddings_train[sd]
        except:
            print('ecception')
            os.remove("%s%s" % (path, self.neg_data_type + "-trainSEUpdated.csv"))
            os.remove("%s%s" % (path, self.neg_data_type + "-trainUpdated.csv"))
            exit(1)
        train_data = deepcopy(train_data_copy)
        return embeddings_train_final.values(), train_data

# # Test data class
# bpdp = True
# if not bpdp:
#     properties_split = ["deathPlace/","birthPlace/","author/","award/","foundationPlace/","spouse/","starring/","subsidiary/"]
#     datasets_class = ["range/","domain/","mix/","property/","domainrange/","random/"]
#     # make it true or false
#     prop_split = True
#     clss = datasets_class
#     if prop_split:
#         clss = properties_split
#
#     for cls in clss:
#         method = "emb-only" #emb-only  hybrid
#         path_dataset_folder = 'dataset/'
#         if prop_split:
#             dataset = Data(data_dir=path_dataset_folder, sub_dataset_path= None, prop = cls)
#         else:
#             dataset = Data(data_dir=path_dataset_folder, sub_dataset_path= cls)
# else:
#     path_dataset_folder = 'dataset/hybrid_data/bpdp/'
#     dataset = Data(data_dir=path_dataset_folder, bpdp_dataset=True)
#     print("success")
