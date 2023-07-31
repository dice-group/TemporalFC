# @staticmethod
#     def get_sent_embeddings(self, path, name, train_data):
#         emb = dict()
#         embeddings_train = dict()
#         # print("%s%s" % (path,name))
#         train_data_copy = deepcopy(train_data)
#         train_data1 = np.array(train_data_copy)
#         i = 0
#         j = 0
#         k = 0
#         jj = 0
#         train_i = 0
#         found = False
#         with open("%s%s" % (path, name), "r") as f:
#             for datapoint in f:
#                 if datapoint.startswith("0\t1\t2"):
#                     continue
#                 else:
#                     if datapoint.startswith("http://dbpedia.org/resource/Vlado_Brankovic"):
#                         print("test")
#                     emb[i] = datapoint.split('\t')
#                     try:
#                         if emb[i][0] != "0":
#                             # remove punctuation from the string
#                             str1 = self.remove_punctuation(emb[i][0])
#                             str2 = self.remove_punctuation(emb[i][1])
#                             str3 = self.remove_punctuation(emb[i][2])
#                             str4 = self.remove_punctuation(train_data1[j][0])
#                             str5 = self.remove_punctuation(train_data1[j][1])
#                             str6 = self.remove_punctuation(train_data1[j][2])
#
#                             if ((str1).__contains__(str4) and
#                                     (str2).__contains__(str5) and
#                                     (str3).__contains__(str6)):
#                                 emb[i][-1] = emb[i][-1].replace("'", "").replace("\n", "")
#                                 if (len(emb[i])) == ((768 * 3) + 3 + 1):
#                                     # because defacto scores are also appended at the end
#                                     embeddings_train[train_i] = emb[i][:-1]
#                                 elif (len(emb[i])) == ((768 * 3) + 3):
#                                     # emb[i][-1] = emb[i]
#                                     embeddings_train[train_i] = emb[i]
#                                 else:
#                                     print("there is something fishy:" + str(emb[i]))
#                                     exit(1)
#                                 # train_data1.__delitem__(train_data1[j])
#                                 train_i += 1
#                                 found = True
#                             else:
#                                 found = False
#                     except:
#                         print('ecception')
#                         os.remove("%s%s" % (path, self.neg_data_type+"-trainSEUpdated.csv"))
#                         os.remove("%s%s" % (path, self.neg_data_type+"-trainUpdated.csv"))
#                         exit(1)
#                     # if (train_i >= 8000):  # len(train_data)
#                     #     break
#                     if found == False:
#                         jj = 0
#                         for t_data in train_data1:
#                             str4 = self.remove_punctuation(t_data[0])
#                             str5 = self.remove_punctuation(t_data[1])
#                             str6 = self.remove_punctuation(t_data[2])
#                             if ((str1).__contains__(str4) and
#                                     (str2).__contains__(str5) and
#                                     (str3).__contains__(str6)):
#                                 emb[i][-1] = emb[i][-1].replace("'", "").replace("\n", "")
#                                 if (len(emb[i])) == ((768 * 3) + 3 + 1):
#                                     # because defacto scores are also appended at the end
#                                     embeddings_train[train_i] = emb[i][:-1]
#                                 elif (len(emb[i])) == ((768 * 3) + 3):
#                                     # emb[i][-1] = emb[i]
#                                     embeddings_train[train_i] = emb[i]
#                                 else:
#                                     print("there is something fishy:" + str(emb[i]))
#                                     exit(1)
#                                 # train_data1.__delitem__(t_data)
#                                 j = jj
#                                 train_i += 1
#                                 found = True
#                                 break
#                             else:
#                                 found = False
#                             jj = jj + 1
#                     if found == False:
#                         # print(emb[i])
#                         if (train_i >= len(train_data)): #len(train_data)
#                             break
#                     i = i + 1
#                     j = j + 1
#                     found = False
#
#                     # i = i+1
#
#         if len(train_data) != len(embeddings_train):
#             print("problem: length of train and sentence embeddings arrays are different:train:"+str(len(train_data))+",emb:"+str(len(embeddings_train)))
#             # exit(1)
#         # following code is just for ordering the data in sentence vectors
#         train_i = 0
#         train_data_copy = deepcopy(train_data)
#         embeddings_train_final = dict()
#         try:
#             for dd in train_data:
#                 found_data = False
#                 jj = 0
#                 for sd in embeddings_train.values():
#                     sub = self.remove_punctuation( dd[0])  # to be updated please
#                     pred = self.remove_punctuation( dd[1])
#                     obj = self.remove_punctuation( dd[2])
#                     sub1 =  self.remove_punctuation( sd[0])
#                     pred1 = self.remove_punctuation( sd[1])
#                     obj1 =  self.remove_punctuation( sd[2])
#
#                     if (((sub == sub1) and (pred == pred1) and (obj == obj1))
#                         or
#                         (( sub1.lower()== sub.lower()) and ( pred1.lower() == pred.lower()) and
#                          ( obj1.lower() == obj.lower()))):
#                         embeddings_train_final[train_i] = sd
#                         train_i+=1
#                         found_data = True
#                         break
#                     jj += 1
#                 if found_data== False:
#                     train_data_copy.remove(dd)
#                     print("missing train data from sentence embeddings file:"+str(dd))
#                 else:
#                     # print("to delete from list: "+str(sd))
#                     embeddings_train =  self.without(embeddings_train,jj)
#                     # embeddings_train = embeddings_train.dropna().reset_index(drop=True)
#                     # del embeddings_train[sd]
#         except:
#             print('ecception')
#             os.remove("%s%s" % (path, self.neg_data_type+"-trainSEUpdated.csv"))
#             os.remove("%s%s" % (path, self.neg_data_type+"-trainUpdated.csv"))
#             exit(1)
#         train_data = deepcopy(train_data_copy)
#         return embeddings_train_final.values(), train_data
