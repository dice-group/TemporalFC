import pandas as pd

from data import Data

import numpy as np
import zipfile


class ConcatEmbeddings:
    def __init__(self,args, path_dataset_folder=None, str2=None, DBpedia34k=False):
        #
        self.args = args
        if DBpedia34k:
            self.concatinateEmbeddingDBpedia34k(self, path_dataset_folder)
        elif str2==None:
            self.concatinateEmbedding(self,path_dataset_folder)
        else:
            self.concatinateEmbeddingsMulticlass(self, path_dataset_folder, str2)

    @staticmethod
    def concatinateEmbeddingDBpedia34k(self, path_dataset_folder=None):
        train_folder = ""
        test_folder = ""
        df_entities = pd.read_csv('Embeddings/ConEx/dbpedia34k/all_entities_embeddings.txt', index_col=0, sep='\t')
        df_relations = pd.read_csv('Embeddings/ConEx/dbpedia34k/all_relations_embeddings.txt', index_col=0, sep='\t')
        df_entities = df_entities.drop_duplicates()
        df_relations = df_relations.drop_duplicates()
        # df_entities = pd.read_csv('Embeddings/TransE/entity_embedding_filter.tsv', index_col=0)
        # df_relations = pd.read_csv('Embeddings/TransE/relations.tsv', index_col=0)
        df_train_sentence = pd.read_csv(path_dataset_folder + 'hybrid_data/train/trainSE.csv', sep='\t')
        df_test_sentence = pd.read_csv(path_dataset_folder + 'hybrid_data/test/testSE.csv', sep='\t')
        dataset1 = Data(self.args)
            # data_dir=path_dataset_folder, subpath="",emb_file="",emb_typ="ConEx", emb_folder="dbpedia34k/")

        train_set = list((dataset1.load_data(path_dataset_folder+'hybrid_data/train/', data_type="train")))
        test_set = list((dataset1.load_data(path_dataset_folder+'hybrid_data/test/', data_type="test")))

        # print(df_entities.loc['M._G._Ramachandran'])
        print(df_train_sentence.head())
        # exit(1)

        # concatinating embeddings of train data
        train_combined_emb_set = []
        for idx, (s, p, o, label) in (enumerate(train_set)):
            try:
                triple_embedding = pd.concat([(df_entities.loc[s]),(df_relations.loc[p]),(df_entities.loc[o])], axis=0)
                # triple_embedding = triple_embedding.loc[~triple_embedding.index.duplicated(keep='first')]
                # triple_embedding.reset_index(inplace=True, drop=True)
                # print(type(triple_embedding))

                # NEED some strategy to remove duplicates
                # if len(train_combined_emb_set)!=0 and \
                #         not (train_combined_emb_set[0].__contains__(s) and
                #     train_combined_emb_set[1].__contains__(p) and
                #     train_combined_emb_set[2].__contains__(o)):
                #     continue
                s = dataset1.update_entity(self, s[1:-1])
                p = dataset1.update_entity(self, p[1:-1])
                o = dataset1.update_entity(self, o[1:-1])
                found = False
                counter = 0
                for val1 in df_train_sentence.values:
                    val1[0] = dataset1.update_entity(self, val1[0])
                    val1[1] = dataset1.update_entity(self, val1[1])
                    val1[2] = dataset1.update_entity(self, val1[2])
                    if (val1[0] == s and val1[1] == p and
                            val1[2] == o):
                        sen_emb = pd.DataFrame(val1[3:])
                        found = True
                        break
                    counter += 1
                if found==False:
                    continue
                else:
                    df_train_sentence = df_train_sentence.drop(counter)
                    df_train_sentence = df_train_sentence.dropna().reset_index(drop=True)

                # print(type(pd.Series(val)))

                #
                # for val2 in df_train_sentence.values:
                #     if val1[0]==val2[0] and val1[1]==val2[1] and val1[2]==val2[2]:
                #         # df_train_sentence.drop(val2)
                #         break
                #     counter+=1


                # df_train_sentence = df_train_sentence.drop(val1)
                triple_sentence_emb = pd.concat([pd.DataFrame([s,p,o]),triple_embedding.T, sen_emb], axis=0)
                com_emb = pd.DataFrame(triple_sentence_emb.T.values)
                com_emb.insert(com_emb.values.size, com_emb.values.size, label)
                print(str(str(idx)+":train:"+str(com_emb)))
                train_combined_emb_set.append(com_emb.values)
                # if idx == 2:
                #     break
            except Exception as e:
                print(e)
                print("train:" + str(idx) + s + "," + p + "," + o + "," + str(label))
                # exit(1)

        # concatinating the embeddings of test data
        test_combined_emb_set = []
        for idx,(s, p, o, label) in (enumerate(test_set)):
            try:
                triple_embedding = pd.concat([(df_entities.loc[s]),(df_relations.loc[p]),(df_entities.loc[o])])
                # print(type(triple_embedding))
                # NEED some strategy to remove duplicates
                # if len(train_combined_emb_set)!=0 and \
                #         not (train_combined_emb_set[0].__contains__(s) and
                #     train_combined_emb_set[1].__contains__(p) and
                #     train_combined_emb_set[2].__contains__(o)):
                #     continue
                s = dataset1.update_entity(self, s[1:-1])
                p = dataset1.update_entity(self, p[1:-1])
                o = dataset1.update_entity(self, o[1:-1])
                found = False
                counter = 0
                for val1 in df_test_sentence.values:
                    val1[0] = dataset1.update_entity(self, val1[0])
                    val1[1] = dataset1.update_entity(self, val1[1])
                    val1[2] = dataset1.update_entity(self, val1[2])
                    if (val1[0] == s and  val1[1] == p and
                             val1[2] == o):
                        sen_emb = pd.DataFrame(val1[3:])
                        found = True
                        break
                    counter += 1
                if found == False:
                    continue
                else:
                    df_test_sentence = df_test_sentence.drop(counter)
                    df_test_sentence = df_test_sentence.dropna().reset_index(drop=True)

                # dropping value from dataframe
                # counter = 0
                # for val2 in df_test_sentence.values:
                #     if val1[0] == val2[0] and val1[1] == val2[1] and val1[2] == val2[2]:
                #         # df_train_sentence.drop(val2)
                #         break
                #     counter += 1
                # df_test_sentence = df_test_sentence.drop(counter)
                # df_test_sentence = df_test_sentence.dropna().reset_index(drop=True)

                triple_sentence_emb = pd.concat([pd.DataFrame([s,p,o]),triple_embedding.T, sen_emb], axis=0)
                com_emb = pd.DataFrame(triple_sentence_emb.T.values)
                com_emb.insert(com_emb.values.size, com_emb.values.size, label)
                print(str(str(idx)+":test:"+str(com_emb)))
                test_combined_emb_set.append(com_emb.values)
                # if idx == 2:
                #     break
            except Exception as e:
                print(e)
                print("test:" + str(idx) + s + "," + p + "," + o + "," + str(label))
                # exit(1)

        print(len(test_set))
        self.saveDataToCSV(train_combined_emb_set, name="trainCombinedEmbeddings", path=path_dataset_folder)
        self.saveDataToCSV(test_combined_emb_set, name="testCombinedEmbeddings", path=path_dataset_folder)
    @staticmethod
    def concatinateEmbeddingsMulticlass(self, path_dataset_folder=None,str2=None):
        train_folder = "hybrid_data/train/"+str2
        test_folder = "hybrid_data/test/"+str2
        print(train_folder)
        df_entities = pd.read_csv('../Embeddings/ConEx/all_entities_embeddings_final.txt', index_col=0)
        # df_entities.drop_duplicate()
        df_relations = pd.read_csv('../Embeddings/ConEx/all_relations_embeddings_final.txt', index_col=0)
        df_entities = df_entities.drop_duplicates()
        df_relations = df_relations.drop_duplicates()
        df_train_sentence = pd.read_csv(path_dataset_folder +train_folder+ 'trainSE.csv')
        df_test_sentence = pd.read_csv(path_dataset_folder + test_folder+ 'testSE.csv')
        dataset1 = Data(self.args)
            # data_dir=path_dataset_folder,subpath=str2)

        train_set = list((dataset1.load_data(path_dataset_folder+train_folder, data_type="train")))
        test_set = list((dataset1.load_data(path_dataset_folder+test_folder, data_type="test")))
        # print(df_entities.loc['M._G._Ramachandran'])
        # print(df_train_sentence.head())
        # exit(1)

        # concatinating embeddings of train data
        train_combined_emb_set = []
        for ((idx, (s, p, o, label)), val) in zip(enumerate(train_set), df_train_sentence.values):
            try:
                # if p == "<http://dbpedia.org/ontology/office>":
                #     p = "<http://dbpedia.org/ontology/leader>"
                triple_embedding = df_entities.loc[s].append(df_relations.loc[p]).append(df_entities.loc[o])
                # print(type(triple_embedding))
                sen_emb = pd.DataFrame(val)
                # print(type(pd.Series(val)))
                triple_sentence_emb = pd.concat([triple_embedding.T, sen_emb], axis=0)
                com_emb = pd.DataFrame(triple_sentence_emb.T.values)
                com_emb.insert(com_emb.values.size, com_emb.values.size, label)
                # print(com_emb)
                train_combined_emb_set.append(com_emb.values)
                # if idx == 2:
                #     break
            except Exception as e:
                print(e)
                print("train:" + str(idx) + s + "," + p + "," + o + "," + str(label))
                # exit(1)

        # concatinating the embeddings of test data
        test_combined_emb_set = []
        for ((idx, (s, p, o, label)), val) in zip(enumerate(test_set), df_test_sentence.values):
            try:
                # if p == "<http://dbpedia.org/ontology/office>":
                #     p = "<http://dbpedia.org/ontology/leader>"
                triple_embedding = df_entities.drop_duplicates().loc[s].append(df_relations.loc[p]).append(df_entities.drop_duplicates().loc[o])
                # print(type(triple_embedding))
                sen_emb = pd.DataFrame(val)
                # print(type(pd.Series(val)))
                triple_sentence_emb = pd.concat([triple_embedding.T, sen_emb], axis=0)
                com_emb = pd.DataFrame(triple_sentence_emb.T.values)
                com_emb.insert(com_emb.values.size, com_emb.values.size, label)
                # print(com_emb)
                test_combined_emb_set.append(com_emb.values)
                # if idx == 2:
                #     break
            except Exception as e:
                print(e)
                print("test:" + str(idx) + s + "," + p + "," + o + "," + str(label))
                # exit(1)

        print(len(test_set))
        self.saveDataToCSV(train_combined_emb_set, "trainCombinedEmbeddings", path_dataset_folder+train_folder)
        self.saveDataToCSV(test_combined_emb_set, "testCombinedEmbeddings", path_dataset_folder+test_folder)


    @staticmethod
    def concatinateEmbedding(self,path_dataset_folder=None):
        train_folder = ""
        test_folder = ""
        df_entities = pd.read_csv('Embeddings/ConEx/all_entities_embeddings_final.txt', index_col=0)
        df_relations = pd.read_csv('Embeddings/ConEx/all_relations_embeddings_final.txt', index_col=0)
        df_entities = df_entities.drop_duplicates()
        df_relations = df_relations.drop_duplicates()
        # df_entities = pd.read_csv('Embeddings/TransE/entity_embedding_filter.tsv', index_col=0)
        # df_relations = pd.read_csv('Embeddings/TransE/relations.tsv', index_col=0)
        df_train_sentence = pd.read_csv(path_dataset_folder + 'train_data/trainSE.csv')
        df_test_sentence = pd.read_csv(path_dataset_folder + 'test_data/testSE.csv')
        dataset1 = Data(self.args)
            # data_dir=path_dataset_folder,subpath=None)

        train_set = list((dataset1.load_data(path_dataset_folder, data_type="train")))
        test_set = list((dataset1.load_data(path_dataset_folder, data_type="test")))
        # print(df_entities.loc['M._G._Ramachandran'])
        print(df_train_sentence.head())
        # exit(1)

        # concatinating embeddings of train data
        train_combined_emb_set = []
        for ((idx, (s, p, o, label)), val) in zip(enumerate(train_set), df_train_sentence.values):
            try:
                triple_embedding = pd.concat([(df_entities.loc[s]),(df_relations.loc[p]),(df_entities.loc[o])])
                # print(type(triple_embedding))
                sen_emb = pd.DataFrame(val)
                # print(type(pd.Series(val)))
                triple_sentence_emb = pd.concat([triple_embedding.T, sen_emb], axis=0)
                com_emb = pd.DataFrame(triple_sentence_emb.T.values)
                com_emb.insert(1068, '1068', label)
                # print(com_emb)
                train_combined_emb_set.append(com_emb.values)
                # if idx == 2:
                #     break
            except Exception as e:
                print(e)
                print("train:" + str(idx) + s + "," + p + "," + o + "," + str(label))
                # exit(1)

        # concatinating the embeddings of test data
        test_combined_emb_set = []
        for ((idx, (s, p, o, label)), val) in zip(enumerate(test_set), df_test_sentence.values):
            try:
                triple_embedding = pd.concat([(df_entities.loc[s]),(df_relations.loc[p]),(df_entities.loc[o])])
                # print(type(triple_embedding))
                sen_emb = pd.DataFrame(val)
                # print(type(pd.Series(val)))
                triple_sentence_emb = pd.concat([triple_embedding.T, sen_emb], axis=0)
                com_emb = pd.DataFrame(triple_sentence_emb.T.values)
                com_emb.insert(1068, '1068', label)
                # print(com_emb)
                test_combined_emb_set.append(com_emb.values)
                # if idx == 2:
                #     break
            except Exception as e:
                print(e)
                print("test:" + str(idx) + s + "," + p + "," + o + "," + str(label))
                # exit(1)

        print(len(test_set))
        self.saveDataToCSV(train_combined_emb_set, name="trainCombinedEmbeddings", path=path_dataset_folder)
        self.saveDataToCSV(test_combined_emb_set, name="testCombinedEmbeddings", path=path_dataset_folder)
    @staticmethod
    def saveDataToCSV(combined_emb_set=None, name="trainCombinedEmbeddings", path = ""):
        X = pd.DataFrame([list(l) for l in combined_emb_set]).stack().apply(pd.Series).reset_index(1, drop=True)
        # X=pd.DataFrame(train_combined_emb_set)
        print(X.head)
        compression_opts = dict(method='zip', archive_name=name+'.csv')
        X.to_csv(path + name+'.zip', index=False, compression=compression_opts,sep='\t')
        with zipfile.ZipFile(path + name+'.zip', 'r') as zip_ref:
            zip_ref.extractall(path)



# DBpedia34kDataset = True
# multiclass = True
# test2 = ["date/","domain/","domainrange/","mix/","property/","random/","range/"]
# path_dataset_folder = 'dataset/'
#
# if DBpedia34kDataset:
#     path_dataset_folder += "complete_dataset/dbpedia34k/"
#     ConcatEmbeddings(path_dataset_folder, DBpedia34k=True)
# elif multiclass:
#     for str22 in test2:
#         ConcatEmbeddings(path_dataset_folder,str22)
# else:
#     ConcatEmbeddings(path_dataset_folder, None)


# print(df_test_sentence.loc[0])
# exit(1)

# train_df=pd.read_csv('dataset/trainSE.csv',index_col=0)
# test_df=pd.read_csv('dataset/testSE.csv',index_col=0)

# df = df.drop_duplicates()



# print(df.loc['John_Adams'])
# dataset = Data(data_dir=path_dataset_folder)

# for idx, train in enumerate(dataset.train_set):
#     print(str(idx) + str(train))




