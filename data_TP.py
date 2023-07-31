from torch.utils.data import DataLoader, random_split
import numpy as np
from copy import deepcopy
import pickle
# from main import argparse_default
import torch
import os
import random
class Data:
    def __init__(self, args=None):
        data_dir = args.path_dataset_folder
        emb_typ = args.emb_type
        selected_dataset_data_dir = data_dir+str(args.eval_dataset).lower()+"/"
        tmp_emb_folder = data_dir + str(args.eval_dataset).lower()+"/temporal_embeddings/"
        property_split = args.prop
        if args.eval_dataset == "Yago3K":
            args.ids_only = True
        ids_only = args.ids_only
        # Quick workaround as we happen to have duplicate triples.
        # None if load complete data, otherwise load parts of dataset with folders in wrong directory.
        # emb_folder = ""
        if ids_only == False:
            if str(args.eval_dataset).lower().__contains__("dbpedia124k"):
                mapped_entities =  self.get_mapped_entities(selected_dataset_data_dir,file_name="map_final")
            if str(args.eval_dataset).lower().__contains__("dbpedia124k") and property_split != "None":
                self.train_set = list((self.load_data(selected_dataset_data_dir + "train/properties/correct/", data_type=property_split)))
                self.test_set = list((self.load_data(selected_dataset_data_dir + "test/properties/correct/", data_type=property_split)))
                self.train_set_time = list((self.load_data_with_time(selected_dataset_data_dir + "train/", data_type="train_original", mapped_entities=mapped_entities,prop=property_split)))
                self.test_set_time = list((self.load_data_with_time(selected_dataset_data_dir + "test/", data_type="test_original", mapped_entities=mapped_entities,prop=property_split)))
                self.train_set_time_final = self.update_and_match_triples_start(self, selected_dataset_data_dir, "train", "properties/correct/"+property_split+"_train_with_time_final.txt",
                                                                                self.train_set, self.train_set_time)
                self.test_set_time_final = self.update_and_match_triples_start(self, selected_dataset_data_dir, "test", "properties/correct/"+property_split+"_test_with_time_final.txt",
                                                                               self.test_set, self.test_set_time)
            elif str(args.eval_dataset).lower().__contains__("dbpedia124k"): #bpdp_dataset == True:
                self.train_set = list((self.load_data(selected_dataset_data_dir+"train/" , data_type="train")))
                self.test_set = list((self.load_data(selected_dataset_data_dir+"test/" , data_type="test")))
                self.train_set_time = list((self.load_data_with_time(selected_dataset_data_dir + "train/", data_type="train_original", mapped_entities =  mapped_entities)))
                self.test_set_time = list((self.load_data_with_time(selected_dataset_data_dir + "test/", data_type="test_original", mapped_entities =  mapped_entities)))
                self.train_set_time_final = self.update_and_match_triples_start(self,selected_dataset_data_dir,"train","train_with_time_final.txt",
                                                                                self.train_set, self.train_set_time)
                self.test_set_time_final = self.update_and_match_triples_start(self,selected_dataset_data_dir,"test","test_with_time_final.txt",
                                                                               self.test_set, self.test_set_time)
            elif str(args.eval_dataset).lower().__contains__("dbpedia5") or str(args.eval_dataset).lower().__contains__("yago3k"):
                self.train_set = list((self.load_data(selected_dataset_data_dir + "train/", data_type="train")))
                self.test_set = list((self.load_data(selected_dataset_data_dir + "test/", data_type="test")))
                self.train_set_time = list((self.load_data_with_time(selected_dataset_data_dir + "train/", data_type="train_original")))
                self.test_set_time = list((self.load_data_with_time(selected_dataset_data_dir + "test/", data_type="test_original")))
                self.train_set_time_final = self.update_and_match_triples_start(self, selected_dataset_data_dir, "train", "train_with_time_final.txt",
                                                                                self.train_set, self.train_set_time)
                self.test_set_time_final = self.update_and_match_triples_start(self, selected_dataset_data_dir, "test", "test_with_time_final.txt",
                                                                               self.test_set, self.test_set_time)

            else:
                print("please specify a valid dataset")
                exit(1)

            split = "" if (property_split == 'None') else "correct/"+property_split
            # factcheck predictions on train and test data
            if str(args.eval_dataset).lower().__contains__("dbpedia124k") and property_split != "None":
                self.train_set_veracity = list(
                    (self.load_data(selected_dataset_data_dir + "factcheck_veracity_scores/properties/train/" + split, data_type="", pred=True)))
                self.test_set_pred = list(
                    (self.load_data(selected_dataset_data_dir + "factcheck_veracity_scores/properties/test/" + split, data_type="", pred=True)))
            elif str(args.eval_dataset).lower().__contains__("dbpedia124k"):
                self.train_set_veracity = list(
                    (self.load_data(selected_dataset_data_dir+"factcheck_veracity_scores/"+split, data_type="train_pred", pred=True)))
                self.test_set_pred = list(
                    (self.load_data(selected_dataset_data_dir+"factcheck_veracity_scores/"+split, data_type="test_pred", pred=True)))

            # self.train_set_time_final = self.update_and_match_triples_start(self,selected_dataset_data_dir,"train","train_with_time_final",self.train_set, self.train_set_time)
            if args.include_veracity == True:
                self.train_set_pred_final = self.update_and_match_triples_start(self,selected_dataset_data_dir,"factcheck_veracity_scores","final_train_pred.txt",
                                                                                self.train_set_veracity, self.train_set_time, properties_split = property_split,veracity=True)
                self.test_set_pred_final = self.update_and_match_triples_start(self,selected_dataset_data_dir,"factcheck_veracity_scores","final_test_pred.txt",
                                                                               self.test_set_pred, self.test_set_time, properties_split = property_split, veracity=True)
                # second time updating time array so no need to swap the sub and objects for negative triples(they are already inside), so veracity = true is fine
                self.train_set_time_final = self.update_match_triples(data_set1=self.train_set_time_final, data_set2=self.train_set_pred_final, veracity=True, final=True)
                self.test_set_time_final = self.update_match_triples(data_set1=self.test_set_time_final, data_set2=self.test_set_pred_final, veracity=True, final=True)
                # check if prediction results and the train and test files are not equal size because we  have to check based on index the veracity scores (opposite check?)
                self.train_set_pred_final = self.check_if_not_equal_size(self.train_set_pred_final,self.train_set_time_final )
                self.test_set_pred_final = self.check_if_not_equal_size(self.test_set_pred_final, self.test_set_time_final)
            # generate test and validation sets
            self.test_set, self.valid_set = self.generate_test_valid_set(self, self.test_set)
            self.test_set_time_final, self.valid_set_time_final = self.generate_test_valid_set(self, self.test_set_time_final)

            if args.include_veracity == True:
                self.test_set_pred, self.valid_set_pred = self.generate_test_valid_set(self, self.test_set_pred)
                self.test_set_pred_final, self.valid_set_pred_final = self.generate_test_valid_set(self, self.test_set_pred_final)

            # get all entities and relations
            # self.data = self.train_set + list(self.test_set) + list(self.valid_set)
            self.data = self.train_set_time + self.test_set_time
            self.entities = self.get_entities(self.data)

                   # self.relations = list(set(self.get_relations(self.train_set) + self.get_relations(self.test_set)))
            self.relations = self.get_relations(self.data)
            self.times = self.get_times(self.data)

            self.idx_ent_dict = dict()
            self.idx_rel_dict = dict()
            self.idx_time_dict = dict()

            # Generate integer mapping
            for i in self.entities:
                self.idx_ent_dict[i.replace("<http://dbpedia.org/resource/", "")[:-1]] = len(self.idx_ent_dict)
            for i in self.relations:
                self.idx_rel_dict[i.replace("<http://dbpedia.org/ontology/", "")[:-1]] = len(self.idx_rel_dict)
            for i in self.times:
                self.idx_time_dict[i] = len(self.idx_time_dict)

            if args.include_veracity == True:
                self.veracity_train = self.get_veracity_data(self, self.train_set_pred_final)
                self.veracity_test = self.get_veracity_data(self, self.test_set_pred_final)
                self.veracity_valid = self.get_veracity_data(self, self.valid_set_pred_final)


            self.emb_entities = self.get_embeddings(tmp_emb_folder+emb_typ+'/','entity.pkl')
            self.emb_relation = self.get_embeddings(tmp_emb_folder+emb_typ+'/','relation.pkl')
            self.emb_time = self.get_embeddings(tmp_emb_folder+emb_typ+'/','time.pkl')

            self.num_entities = len(self.emb_entities)
            self.num_relations = len(self.emb_relation)
            self.num_times = len(self.emb_time)

            if args.negative_triple_generation =="corrupted-time-based":
                self.train_set_time_final = self.generate_negative_triples(self.train_set_time_final)
                self.valid_set_time_final = self.generate_negative_triples(self.valid_set_time_final)
                self.test_set_time_final = self.generate_negative_triples(self.test_set_time_final)
            elif args.negative_triple_generation == "False":
                self.train_set_time_final = self.generate_only_true_triples(self.train_set_time_final)
                self.valid_set_time_final = self.generate_only_true_triples(self.valid_set_time_final)
                self.test_set_time_final = self.generate_only_true_triples(self.test_set_time_final)
            self.idx_train_set = []
            i = 0
            len_train = len(self.train_set_time_final)
            for (s, p, o, time, label) in self.train_set_time_final:
                s = str(s).replace("<http://dbpedia.org/resource/", "")[:-1]
                p = str(p).replace("<http://dbpedia.org/ontology/", "")[:-1].replace("Of","")
                o = str(o).replace("<http://dbpedia.org/resource/", "")[:-1]
                if self.idx_ent_dict.keys().__contains__(s) and self.idx_rel_dict.keys().__contains__(p) and self.idx_ent_dict.keys().__contains__(o):
                    idx_s, idx_p, idx_o, idx_t, label = self.idx_ent_dict[s], self.idx_rel_dict[p], self.idx_ent_dict[o], self.idx_time_dict[time], label
                    if label == 'True':
                        label = 1
                    else:
                        label = 0
                    ver = i
                    #     this is to just to make sure if any time is not the same even after randomaly shuffling so increment by 1
                    if args.negative_triple_generation =="corrupted-time-based" and (i >= int(len_train/2)):
                        item = self.idx_train_set.__getitem__(i-int(len_train/2))
                        if ((item[0] != int(idx_s)) or (item[1] != int(idx_p)) or (item[2]!=int(idx_o))):
                            print("serious problem, please check")
                            exit(1)
                        if ((item[0]==int(idx_s)) and (item[1]==int(idx_p)) and (item[2]==int(idx_o)) and (item[3]==int(idx_t))):
                            idx_t = (int(idx_t) + 1) if ((int(idx_t)+1) < self.num_times) else 0
                        ver = item[4]
                    self.idx_train_set.append([int(idx_s), int(idx_p), int(idx_o),int(idx_t), ver, label])
                else:
                    print("check:"+s + ","+o)
                i = i + 1

            self.idx_valid_set = []
            j = 0
            len_valid = len(self.valid_set_time_final)
            for (s, p, o, time, label) in self.valid_set_time_final:
                s = str(s).replace("<http://dbpedia.org/resource/", "")[:-1]
                p = str(p).replace("<http://dbpedia.org/ontology/", "")[:-1].replace("Of","")
                o = str(o).replace("<http://dbpedia.org/resource/", "")[:-1]
                if self.idx_ent_dict.keys().__contains__(s) and  self.idx_rel_dict.keys().__contains__(p) and self.idx_ent_dict.keys().__contains__(o):
                    idx_s, idx_p, idx_o, idx_t, label = self.idx_ent_dict[s], self.idx_rel_dict[p], self.idx_ent_dict[o],self.idx_time_dict[time], label
                    if label == 'True':
                        label = 1
                    else:
                        label = 0
                    ver = j
                    #     this is to check if any time is same even after randomaly shuffling so increment by 1
                    if args.negative_triple_generation =="corrupted-time-based" and (j >= int(len_valid/2)):
                        item = self.idx_valid_set.__getitem__(j-int(len_valid/2))
                        if ((item[0] != int(idx_s)) or (item[1] != int(idx_p)) or (item[2]!=int(idx_o))):
                            print("serious problem, please check")
                            exit(1)
                        if ((item[0]==int(idx_s)) and (item[1]==int(idx_p)) and (item[2]==int(idx_o)) and (item[3]==int(idx_t))):
                            idx_t = (int(idx_t) + 1) if ((int(idx_t)+1) < self.num_times) else 0
                        ver = item[4]
                    self.idx_valid_set.append([int(idx_s), int(idx_p), int(idx_o),int(idx_t), ver, label])
                else:
                    print("check:" + s + "," + o)
                j = j + 1

            self.idx_test_set = []
            k = 0
            len_test = len(self.test_set_time_final)
            for (s, p, o, time, label) in self.test_set_time_final:
                s = str(s).replace("<http://dbpedia.org/resource/", "")[:-1]
                p = str(p).replace("<http://dbpedia.org/ontology/", "")[:-1].replace("Of","")
                o = str(o).replace("<http://dbpedia.org/resource/", "")[:-1]
                if self.idx_ent_dict.keys().__contains__(s) and  self.idx_rel_dict.keys().__contains__(p) and self.idx_ent_dict.keys().__contains__(o):
                    idx_s, idx_p, idx_o, idx_t, label = self.idx_ent_dict[s], self.idx_rel_dict[p], self.idx_ent_dict[o],self.idx_time_dict[time], label
                    if label == 'True':
                        label = 1
                    else:
                        label = 0
                    ver = k
                    #     this is to check if any time is same even after randomaly shuffling so increment by 1
                    if args.negative_triple_generation =="corrupted-time-based" and (k >= int(len_test/2)):
                        item = self.idx_test_set.__getitem__(k-int(len_test/2))
                        if ((item[0] != int(idx_s)) or (item[1] != int(idx_p)) or (item[2]!=int(idx_o))):
                            print("serious problem, please check")
                            exit(1)
                        if ((item[0]==int(idx_s)) and (item[1]==int(idx_p)) and (item[2]==int(idx_o)) and (item[3]==int(idx_t))):
                            idx_t = (int(idx_t) + 1) if ((int(idx_t)+1) < self.num_times) else 0
                        ver = item[4]
                    self.idx_test_set.append([int(idx_s), int(idx_p), int(idx_o),int(idx_t), ver, label])
                else:
                    print("check:" + s + "," + o)
                k = k + 1
        else:
            # self.idx_ent_dict = self.get_ids_dict(selected_dataset_data_dir+"entities")
            # self.idx_rel_dict = self.get_ids_dict(selected_dataset_data_dir+"relations")
            # self.idx_time_dict = self.get_ids_dict(selected_dataset_data_dir+"times")
            self.emb_entities = self.get_embeddings( tmp_emb_folder + emb_typ + '/', 'entity.pkl')
            self.emb_relation = self.get_embeddings( tmp_emb_folder + emb_typ + '/', 'relation.pkl')
            self.emb_time = self.get_embeddings( tmp_emb_folder + emb_typ + '/', 'time.pkl')
            self.num_entities = len(self.emb_entities)
            self.num_relations = len(self.emb_relation)
            self.num_times = len(self.emb_time)
            self.idx_train_set = self.get_ids_dict(selected_dataset_data_dir+"train/train")
            self.idx_test_set = self.get_ids_dict(selected_dataset_data_dir+"test/test")
            self.idx_valid_set = self.get_ids_dict(selected_dataset_data_dir+"test/valid")
    def generate_negative_triples(self, data):
        data2 = []
        data_final = []
        i =0
        times = []
        for (s, p, o, time, label) in data:
            if label == 'True':
                times.append(time)
                data2.append([s, p, o, time, label])
            i = i + 1
        random.shuffle(times)
        data3 = []
        for j in range(len(data2)):
            item = data2.__getitem__(j)
            tim = times.__getitem__(j)
            data3.append([item[0],item[1],item[2],tim,'False'])

        data_final = data2 + data3
        # data_final.append(data3)
        return data_final

    def generate_only_true_triples(self, data):
        data2 = []
        data_final = []
        i =0
        times = []
        for (s, p, o, time, label) in data:
            if label == 'True\n':
                label = label[:-1]
            if label == 'True':
                times.append(time)
                data_final.append([s, p, o, time, label])
            i = i + 1

        # data_final.append(data3)
        return data_final

    @staticmethod
    def get_veracity_data(self, train_emb):
        embeddings_train = dict()
        i = 0
        for train in train_emb:
            embeddings_train[i] = train[3]
            i += 1

        return embeddings_train.values()
    @staticmethod
    def update_and_match_triples_start(self, selected_dataset_data_dir, type, file_name, data_set1, data_set2,  properties_split = None, veracity = False):
        if veracity==False:
            if (os.path.exists(selected_dataset_data_dir + type+ "/"+ file_name)):
                self.set_time_final = list(self.load_data(selected_dataset_data_dir + type+"/", data_type=str(file_name).replace(".txt",""),pred=True))
            else:
                if len(data_set1) != len(data_set2):
                    self.set_time_final = self.update_match_triples(data_set1, data_set2)
                else:
                    self.set_time_final = data_set2
                self.save_triples(selected_dataset_data_dir, type+"/"+file_name, self.set_time_final)
        else:
            tt = "properties/train/" if (file_name.__contains__("train")) else "properties/test/"
            split = "" if (properties_split=='None') else tt+"correct/" +properties_split + "_"
            if (os.path.exists(selected_dataset_data_dir + type+ "/"+ split+ file_name)):
                self.set_time_final = list(self.load_data(selected_dataset_data_dir + type+"/"+split , data_type=str(file_name).replace(".txt",""),pred=True))
            else:
                self.set_time_final = self.update_match_triples(data_set1, data_set2, veracity=veracity)
                self.save_triples(selected_dataset_data_dir, type+"/"+split+file_name, self.set_time_final,veracity=veracity)
        return self.set_time_final

    def is_valid_test_available(self):
        if len(self.idx_valid_set) > 0 and len(self.idx_test_set) > 0:
            return True
        return False

    # @staticmethod
    # def load_triples(data_dir, type, triples):
    #     with open(data_dir + type + '.txt', "r") as f:
    #         for item in triples:
    #             f.write("%s\n" % item)
    @staticmethod
    def save_triples(data_dir,type, triples,veracity=False):
        if veracity==False:
            with open(data_dir + type, "w") as f:
                for item in triples:
                    f.write(""+(item[0])+"\t"+(item[1])+"\t"+(item[2])+"\t"+str(item[3])+"\t"+str(item[4])+"\n")
        else:
            with open(data_dir + type, "w") as f:
                for item in triples:
                    f.write(""+str(item[0])+"\t"+str(item[1])+"\t"+str(item[2])+"\t"+str(item[3])+"\n")
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
    def generate_test_valid_set(self, test_set):
        test_data = []
        valid_data = []
        i = 0
        sent_i = 0
        for data in test_set:
            if i % 20 == 0:
                valid_data.append(data)
            else:
                test_data.append(data)

            i += 1
        return  test_data, valid_data
    def get_ids_dict(self, dict_file_path):
        ids_dict = dict()
        data = []
        with open("%s" % (dict_file_path), "r") as f:
            for datapoint in f:
                datapoint = datapoint.split()
                if len(datapoint) == 2:
                    ids_dict[datapoint[1]] = datapoint[0]
                elif len(datapoint)==5:
                    arr = []
                    for tt in datapoint:
                        arr.append(int(tt))
                    arr.append(True)
                    data.append(arr)
                else:
                    print("invalid format")
                    exit(1)
        if len(ids_dict) > 0:
            return ids_dict
        else:
            return data
    @staticmethod
    def load_data(data_dir, data_type, pred=False):
        try:
            data = []
            if pred == False:
                with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                    for datapoint in f:
                        datapoint = datapoint.split("\t")
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
                            s, p, o, time, label = datapoint
                            assert label == 'True' or label == 'False'
                            if label == 'True':
                                label = 1
                            else:
                                label = 0
                            data.append((s, p, o, time, label))
                        else:
                            raise ValueError
            else:
                with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                    for datapoint in f:
                        datapoint = datapoint.split('\t')
                        if len(datapoint) == 4:
                            s, p, o, label = datapoint
                            data.append((s, p, o, label))
                        elif len(datapoint) == 3:
                            s, p, label = datapoint
                            data.append((s, p, 'DUMMY', label))
                        elif len(datapoint) == 5:
                            s, p, o, time, label = datapoint
                            if label=="." and str(time).__contains__("^<http://www.w3.org/2001/XMLSchema#double>"):
                                label = time
                                label = str(label).replace("\"^^<http://www.w3.org/2001/XMLSchema#double>", "")
                                data.append((s, p, o, label))
                            else:
                                data.append((s, p, o, time, label))
                        else:
                            raise ValueError
        except FileNotFoundError as e:
            print(e)
            print('Add empty.')
            data = []
        return data
    @staticmethod
    def get_mapped_entities(data_dir, file_name):
        mapping_entities = dict()
        with open("%s%s.txt" % (data_dir, file_name), "r") as f:
            for datapoint in f:
                datapoint = datapoint.split("	->	")
                if len(datapoint) == 2:
                    mapping_entities[datapoint[0]] = datapoint[1].replace("\n","")
        return mapping_entities

    def check_if_not_equal_size(self, data_set1, data_set2):
        data = []
        if len(data_set1)!=len(data_set2):
            train_set_time1 = deepcopy(data_set1)
            for tp in data_set2:
                found = False
                for tpt in train_set_time1:
                    if (tp[0] == tpt[0] and tp[1] == tpt[1].replace("Of", "") and tp[2] == tpt[2]):
                        data.append([tpt[0], tpt[1], tpt[2], tpt[3]])
                        found = True
                        break
                    elif (tp[2] == tpt[0] and tp[0] == tpt[2]):
                        data.append([tpt[0], tpt[1], tpt[2], tpt[3]])
                        found = True
                        break
                if found == False:
                    print("Embeddings not found: excluded triple:" + str(tp))
                else:
                    train_set_time1.remove(tpt)

            return data
        else:
            return data_set1

    # update date set 1 and match with dataset 2
    @staticmethod
    def update_match_triples(data_set1, data_set2, veracity=False, final= False):
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
                    if (tp[0] == tpt[0] and tp[1] == tpt[1].replace("Of","") and tp[2] == tpt[2]):
                        data.append([tp[0],tp[1],tp[2],tpt[3],tpt[4]])
                        found = True
                        break
                    elif(tp[2] == tpt[0] and tp[0] == tpt[2]):# to cover negative triples we are doing like this, swaping the sub and obj and not checking the predicate
                        data.append([tpt[0], tpt[1], tpt[2], tpt[3], 'False'])
                        found = True
                        break

                if found == False:
                    print("not found:"+ str(tp))
            elif veracity==True and final==True: # final is for second check
                for tpt in data_set21:
                    if (tp[0] == tpt[0] and tp[1] == tpt[1].replace("Of", "") and tp[2] == tpt[2]):
                        data.append([tp[0], tp[1], tp[2], tp[3], tp[4]])
                        found = True
                        break
                    elif (tp[2] == tpt[0] and tp[1] == tpt[1].replace("Of", "") and tp[0] == tpt[2]):# to cover negative triples we are doing like this, swaping the sub and obj and not checking the predicate
                        data.append([tp[0], tp[1], tp[2], tp[3], tp[4]])
                        found = True
                        break
                if found == False:
                    print("not found:"+ str(tp))
                else:
                    data_set21.remove(tpt)

            else:
                for tpt in data_set21:
                    if (tp[0] == tpt[0] and tp[1] == tpt[1].replace("Of", "") and tp[2] == tpt[2]):
                        data.append([tp[0], tp[1], tp[2], tp[3]])
                        found = True
                        break
                    elif (tp[2] == tpt[0] and tp[0] == tpt[2]): # to cover negative triples we are doing like this, swaping the sub and obj and not checking the predicate
                        data.append([tp[0], tp[1], tp[2], tp[3]])
                        found = True
                        break
                if found == False:
                    print("not found:"+ str(tp))
                # break

                # else:
                #     print("problematic triple:"+ str(tp))

        return data

    @staticmethod
    def load_data_with_time(data_dir, data_type, mapped_entities=None, prop = None):
        try:
            data = []
            with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                for datapoint in f:
                    datapoint = datapoint.split("\t")
                    if len(datapoint) >= 5:
                        if len(datapoint) >5:
                            datapoint[5] = '_'.join(datapoint[4:])
                        s, p, o, time, loc = datapoint[0:5]
                        if prop!=None:
                            if not str(p).__eq__(prop+"Of"):
                                continue
                        s = "http://dbpedia.org/resource/" + s
                        if (mapped_entities!=None and s in mapped_entities.keys()):
                            s = mapped_entities[s]
                        p = "http://dbpedia.org/ontology/" + p
                        o = "http://dbpedia.org/resource/" + o
                        if (mapped_entities!=None and o in mapped_entities.keys()):
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
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities

    @staticmethod
    def get_times(data):
        times = sorted(list(set([d[3] for d in data])))
        return times
    # / home / umair / Documents / pythonProjects / HybridFactChecking / Embeddings / ConEx_dbpedia
    @staticmethod
    def get_embeddings(path,name):
        # embeddings = dict()
        # print("%s%s.txt" % (path,name))
        embd = torch.load("%s%s" % (path,name),map_location=torch.device('cpu'))
        # old_data = pickle.load(file)
        # with open("%s%s" % (path,name), 'rb') as f:
        #     data = pickle.load(f)
        # with open("%s%s" % (path,name), "r") as f:
        #     for datapoint in f:
        #         if datapoint.startswith("0	 1	 2	 3	 4	 5	 6	 7	 8"):
        #             continue
        #         data = datapoint.split('>\t')
        #         if len(data)==1:
        #             print("stoped: getting embeddings function")
        #             exit(1)
        #         if len(data) > 1:
        #             data2 = data[0]+">",data[1].split('\t')
        #             # test = data2[0].replace("\"","").replace("_com",".com").replace("Will-i-am","Will.i.am").replace("Will_i_am","Will.i.am")
        #             # test = data2[0].replace("\"","")
        #             if data2[0] in idxs:
        #                 embeddings[data2[0]] = data2[1]
        # for emb in idxs:
        #     if emb not in embeddings.keys():
        #         print("this is missing in embeddings file:"+ emb)
        #         exit(1)
        #
        # if len(idxs) > len(embeddings):
        #     print("embeddings missing")
        #     exit(1)
        # embeddings_final = dict()
        # for emb in idxs.keys():
        #     if emb in embeddings.keys():
        #         embeddings_final[emb] = embeddings[emb]
        #     else:
        #         print('no embedding', emb)
        #         exit(1)

        return embd.weight

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
    def get_copaal_veracity(path, name, train_set):
        emb = dict()

        embeddings_train = dict()
        # print("%s%s" % (path,name))

        i = 0
        train_i = 0
        found = False
        with open("%s%s" % (path, name), "r") as f:
            for datapoint in f:
                if datapoint.startswith("0,1,2"):
                    continue
                else:
                    emb[i] = datapoint.split(',')
                    try:
                        for dd in train_set:
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
                        if (train_i >= len(train_set)):
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
            for embb in train_set:
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

        if len(train_set) != len(embeddings_train_final):
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
        return ent

    def without(self,d, key):
        new_d = d.copy()
        new_d2 = dict()
        new_d.pop(key)
        count = 0
        for dd in new_d.values():
            new_d2[count]=dd
            count+=1
        return new_d2
    @staticmethod
    def get_sent_embeddings(self, path, name, train_set):
        emb = dict()

        embeddings_train = dict()
        # print("%s%s" % (path,name))
        train_set_copy = deepcopy(train_set)
        i = 0
        train_i = 0
        found = False
        with open("%s%s" % (path, name), "r") as f:
            for datapoint in f:
                if datapoint.startswith("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"):
                    continue
                else:
                    if datapoint.startswith("http://dbpedia.org/resource/Vlado_Brankovic"):
                        print("test")
                    emb[i] = datapoint.split('\t')
                    try:
                        if emb[i][0] != "0":
                            for dd in train_set_copy:
                                # updated because factcheck results does not contained punctuations
                                sub = self.update_entity(self, dd[0])
                                pred = self.update_entity(self, dd[1])
                                obj = self.update_entity(self, dd[2])

                                emb[i][0] = self.update_entity(self, emb[i][0])
                                emb[i][1] = self.update_entity(self, emb[i][1])
                                emb[i][2] = self.update_entity(self, emb[i][2])

                                if ((emb[i][0] == sub) and
                                        (emb[i][1] == pred) and
                                        (emb[i][2] == obj)
                                        or
                                        ('<'+emb[i][0].lower()+'>' == sub.lower()) and
                                        ('<'+emb[i][1].lower()+'>' == pred.lower()) and
                                        ('<'+emb[i][2].lower()+'>' == obj.lower())):
                                    # print('train data found')
                                    emb[i][-1] = emb[i][-1].replace("'", "").replace("\n","")
                                    if (len(emb[i])) == ((768 * 3) +3 + 1):
                                        # because defacto scores are also appended at the end
                                        embeddings_train[train_i] = emb[i][:-1]
                                    elif (len(emb[i])) == ((768 * 3) +3):
                                        # emb[i][-1] = emb[i]
                                        embeddings_train[train_i] = emb[i]
                                    else:
                                        print("there is something fishy:"+str(emb[i]))
                                        exit(1)

                                    train_i += 1
                                    found = True
                                    break

                                # else:
                                #     print('error')
                                # exit(1)
                    except:
                        print('ecception')
                        exit(1)
                    if found==True:
                        train_set_copy.remove(dd)
                    # else:
                    #     train_set.remove(dd)
                    if found == False:
                        if (train_i >= len(train_set)):
                            break
                        else:
                            print("some training data missing....not found:" + str(emb[i]))
                            print(i)
                            print("test")
                            # exit(1)
                    i = i + 1
                    found = False

                    # i = i+1

        if len(train_set) != len(embeddings_train):
            print("problem: length of train and sentence embeddings arrays are different:train:"+str(len(train_set))+",emb:"+str(len(embeddings_train)))
            # exit(1)
        # following code is just for ordering the data in sentence vectors
        train_i = 0
        train_set_copy = deepcopy(train_set)
        embeddings_train_final = dict()
        for dd in train_set:
            found_data = False
            jj = 0
            for sd in embeddings_train.values():
                sub = self.update_entity(self, dd[0])
                pred =self.update_entity(self, dd[1])
                obj = self.update_entity(self, dd[2])
                sub1 = '<'+self.update_entity(self, sd[0])+'>'
                pred1 = '<'+self.update_entity(self, sd[1])+'>'
                obj1 = '<'+self.update_entity(self, sd[2])+'>'
                if (((sub == sub1) and (pred == pred1) and (obj == obj1))
                    or
                    (( sub1.lower()== sub.lower()) and ( pred1.lower() == pred.lower()) and
                     ( obj1.lower() == obj.lower()))):
                    embeddings_train_final[train_i] = sd
                    train_i+=1
                    found_data = True
                    break
                jj += 1
            if found_data== False:
                train_set_copy.remove(dd)
                print("missing train data from sentence embeddings file:"+str(dd))
            else:
                # print("to delete from list: "+str(sd))
                embeddings_train =  self.without(embeddings_train,jj)
                # embeddings_train = embeddings_train.dropna().reset_index(drop=True)
                # del embeddings_train[sd]

        train_set = deepcopy(train_set_copy)
        return embeddings_train_final.values(), train_set


    @staticmethod
    def update_copaal_veracity_score(self, train_emb):
        embeddings_train = dict()
        i = 0
        for train in train_emb:
            embeddings_train[i] = train[3:]
            i += 1

        return embeddings_train.values()

    @staticmethod
    def update_veracity_train_set(self, train_emb):
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
        for train in train_emb:
            embeddings_train[i] = train[3:]
            i+=1

        return embeddings_train.values()

    @staticmethod
    def get_veracity_test_valid_set(path, name, test_set, valid_set):
        embeddings_test, embeddings_valid = dict(), dict()
        emb = dict()
        # print("%s%s" % (path, name))
        found = False
        i = 0
        test_i = 0
        valid_i = 0
        with open("%s%s" % (path, name), "r") as f:
            for datapoint in f:
                if datapoint.startswith("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"):
                    continue
                else:
                    emb[i] = datapoint.split(',')
                    try:
                        for dd in test_set:
                            # figure out some way to handle this first argument well
                            if (emb[i][0].replace(',', '') == dd[0].replace(',', '')) and (
                                    emb[i][1].replace(',', '') == dd[1].replace(',', '')) and (
                                    emb[i][2].replace(',', '') == dd[2].replace(',', '')):
                                # print('test data found')
                                embeddings_test[test_i] = np.append(emb[i][:3],emb[i][-1].replace("\n",""))
                                test_i += 1
                                found = True
                                break
                        for vd in valid_set:
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
        for dd in test_set:
            for et in embeddings_test.values():
                if (et[0].replace(',', '') == dd[0].replace(',', '')) and (
                        et[1].replace(',', '') == dd[1].replace(',', '')) and (
                        et[2].replace(',', '') == dd[2].replace(',', '')):
                    embeddings_test_final[i] = et
                    i = i + 1
                    break
        i = 0
        for dd in valid_set:
            # print(dd)
            for et in embeddings_valid.values():
                if (et[0].replace(',', '') == dd[0].replace(',', '')) and (
                        et[1].replace(',', '') == dd[1].replace(',', '')) and (
                        et[2].replace(',', '') == dd[2].replace(',', '')):
                    embeddings_valid_final[i] = et
                    i = i + 1
                    break
        if (len(embeddings_valid_final) != len(valid_set)) and (len(embeddings_test_final) != len(test_set)):
            exit(1)
        return embeddings_test_final.values(), embeddings_valid_final.values()


    @staticmethod
    def get_sent_test_valid_embeddings(self, path, name, test_set, valid_set):
        embeddings_test, embeddings_valid = dict(),dict()
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
                        if emb[i][0] != "\"0\"":
                            for dd in test_set:
                                # figure out some way to handle this first argument well
                                sub = self.update_entity(self, dd[0])
                                pred = self.update_entity(self, dd[1])
                                obj = self.update_entity(self, dd[2])

                                emb[i][0] = self.update_entity(self, emb[i][0])
                                emb[i][1] = self.update_entity(self, emb[i][1])
                                emb[i][2] = self.update_entity(self, emb[i][2])


                                if  (((emb[i][0].replace(',', '') == sub.replace(',','')) and
                                        (emb[i][1].replace(',', '') == pred.replace(',','')) and (
                                        emb[i][2].replace(',', '') == obj.replace(',','')))
                                        or
                                        (('<'+emb[i][0].lower()+'>' == sub.lower()) and
                                        ('<'+emb[i][1].lower()+'>' == pred.lower()) and
                                        ('<'+emb[i][2].lower()+'>' == obj.lower()))):
                                    # print('test data found')
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
                                    break
                            if found == False:
                                for vd in valid_set:
                                    sub = self.update_entity(self, vd[0])
                                    pred = self.update_entity(self, vd[1])
                                    obj = self.update_entity(self, vd[2])

                                    emb[i][0] = self.update_entity(self, emb[i][0])
                                    emb[i][1] = self.update_entity(self, emb[i][1])
                                    emb[i][2] = self.update_entity(self, emb[i][2])

                                    # figure out some way to handle this first argument well
                                    if (((emb[i][0].replace(',', '') == sub.replace(',', '')) and (
                                            emb[i][1].replace(',', '') == pred.replace(',', '')) and (
                                            emb[i][2].replace(',', '') == obj.replace(',', '')))
                                            or
                                            (('<' + emb[i][0].lower() + '>' == sub.lower()) and
                                             ('<' + emb[i][1].lower() + '>' == pred.lower()) and
                                             ('<' + emb[i][2].lower() + '>' == obj.lower()))):
                                        # print('valid data found')
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
                                        # embeddings_valid[valid_i] = emb[i]
                                        valid_i += 1
                                        found = True
                                        break
                            if found == False:
                                print("some data missing from test and validation sets..error"+ str(emb[i]))
                                    # exit(1)
                            else:

                                found = False

                    except:
                        print('ecception')
                        exit(1)
                    i = i + 1

        # embeddings_test_final, embeddings_valid_final = dict(), dict()
        # i = 0
        # for dd in test_set:
        #     for et in embeddings_test.values():
        #         if ((et[0].replace(',', '') == dd[0].replace(',', '')) and \
        #                 (et[1].replace(',', '') == dd[1].replace(',', '')) and \
        #                 (et[2].replace(',', '') == dd[2].replace(',', '')) \
        #                 or
        #                 (('<' + et[0].lower() + '>' == dd[0].lower()) and
        #                  ('<' + et[1].lower() + '>' == dd[1].lower()) and
        #                  ('<' + et[2].lower() + '>' == dd[2].lower()))):
        #             embeddings_test_final[i] = et
        #             i = i + 1
        #             break
        # i = 0
        # for dd in valid_set:
        #     # print(dd)
        #     for et in embeddings_valid.values():
        #         if ((et[0].replace(',', '') == dd[0].replace(',', '')) and\
        #                 (et[1].replace(',', '') == dd[1].replace(',', '')) and\
        #                 (et[2].replace(',', '') == dd[2].replace(',', ''))
        #                 or
        #                 (('<' + et[0].lower() + '>' == dd[0].lower()) and
        #                  ('<' + et[1].lower() + '>' == dd[1].lower()) and
        #                  ('<' + et[2].lower() + '>' == dd[2].lower()))):
        #             embeddings_valid_final[i] = et
        #             i = i + 1
        #             break
        if (len(embeddings_valid)!= len(valid_set)) and (len(embeddings_test)!= len(test_set)):
            print("check lengths of valid and test data:valid_emb:"+str(len(embeddings_valid))+
                  " valid_set"+str(len(valid_set))+
                  "test_set:"+str(len(test_set))+"test_emb:"+str(len(embeddings_test)))
            # exit(1)
        train_i = 0
        test_set_copy = deepcopy(test_set)
        valid_set_copy = deepcopy(valid_set)
        embeddings_test_final = dict()
        embeddings_valid_final = dict()
        for dd in test_set:
            found_data = False
            jj = 0
            for sd in embeddings_test.values():
                sub = self.update_entity(self, dd[0])
                pred = self.update_entity(self, dd[1])
                obj = self.update_entity(self, dd[2])
                sub1 = '<' + self.update_entity(self, sd[0]) + '>'
                pred1 = '<' + self.update_entity(self, sd[1]) + '>'
                obj1 = '<' + self.update_entity(self, sd[2]) + '>'
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
                test_set_copy.remove(dd)
                print("missing test data from sentence embeddings file:" + str(dd))
            else:
                # embeddings_test.pop(jj)
                embeddings_test = self.without(embeddings_test, jj)
                # print("to delete from list: " + str(sd))
                # del embeddings_test[sd]

        test_set = deepcopy(test_set_copy)

        train_i = 0
        for dd in valid_set:
            found_data = False
            jj = 0
            for sd in embeddings_valid.values():
                sub = self.update_entity(self, dd[0])
                pred = self.update_entity(self, dd[1])
                obj = self.update_entity(self, dd[2])
                sub1 = '<' + self.update_entity(self, sd[0]) + '>'
                pred1 = '<' + self.update_entity(self, sd[1]) + '>'
                obj1 = '<' + self.update_entity(self, sd[2]) + '>'
                if (((sub == sub1) and (pred == pred1) and (obj == obj1))
                        or
                        ((sub1.lower() == sub.lower()) and (pred1.lower() == pred.lower()) and
                         (obj1.lower() == obj.lower()))):
                    embeddings_valid_final[train_i] = sd
                    train_i += 1
                    found_data = True
                    break
                jj += 1
            if found_data == False:
                valid_set_copy.remove(dd)
                print("missing valid data from sentence embeddings file:" + str(dd))
            else:
                # print("to delete from list: " + str(sd))
                # embeddings_valid.pop(jj)
                embeddings_valid = self.without(embeddings_valid, jj)


        valid_set = deepcopy(valid_set_copy)

        return embeddings_test_final.values(), embeddings_valid_final.values(), test_set, valid_set


        # return embeddings.values()


# args = argparse_default()
# dataset = Data(args=args)

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
