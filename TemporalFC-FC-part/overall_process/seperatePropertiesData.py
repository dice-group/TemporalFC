# from GenerateTrainTestTripleSet import load_data
class SeperatePropertiesData:

    def __init__(self,data_dir=None, multiclass=True):
        # splitting test data
        training_data = self.load_data(data_dir + "complete_dataset/", "train")
        testing_data = self.load_data(data_dir + "complete_dataset/", "test")
        # loading predictions
        training_data_pred = self.load_data(data_dir + "complete_dataset/", "train_pred", True)
        testing_data_pred = self.load_data(data_dir + "complete_dataset/", "test_pred", True)

        train_deathplace_prop = []
        train_award_prop = []
        train_subsidiary_prop = []
        train_author_prop = []
        train_starring_prop = []
        train_foundationPlace_prop = []
        train_spouse_prop = []
        train_birthPlace_prop = []
        test_deathplace_prop = []
        test_award_prop = []
        test_subsidiary_prop = []
        test_author_prop = []
        test_starring_prop = []
        test_foundationPlace_prop = []
        test_spouse_prop = []
        test_birthPlace_prop = []


        count = 0
        count1 = 0
        for line in training_data:
            count = count +1
            if str(line[1]).__eq__('<http://dbpedia.org/ontology/deathPlace>'):
                train_deathplace_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/award>'):
                train_award_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/subsidiary>'):
                train_subsidiary_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/author>'):
                train_author_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/starring>'):
                train_starring_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/spouse>'):
                train_spouse_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/foundationPlace>'):
                train_foundationPlace_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/birthPlace>'):
                train_birthPlace_prop.append(line)
            else:
                count = count - 1
                exit(1)



        for line in testing_data:
            count1 = count1 +1
            if str(line[1]).__eq__('<http://dbpedia.org/ontology/deathPlace>'):
                test_deathplace_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/award>'):
                test_award_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/subsidiary>'):
                test_subsidiary_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/author>'):
                test_author_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/starring>'):
                test_starring_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/spouse>'):
                test_spouse_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/foundationPlace>'):
                test_foundationPlace_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/birthPlace>'):
                test_birthPlace_prop.append(line)
            else:
                count1 = count1 - 1
                exit(1)

        self.save_data(self, data_dir, "train/deathPlace/", "train", train_deathplace_prop)
        self.save_data(self, data_dir, "train/award/", "train", train_award_prop)
        self.save_data(self, data_dir, "train/subsidiary/", "train", train_subsidiary_prop)
        self.save_data(self, data_dir, "train/author/", "train", train_author_prop)
        self.save_data(self, data_dir, "train/starring/", "train", train_starring_prop)
        self.save_data(self, data_dir, "train/spouse/", "train", train_spouse_prop)
        self.save_data(self, data_dir, "train/foundationPlace/", "train", train_foundationPlace_prop)
        self.save_data(self, data_dir, "train/birthPlace/", "train", train_birthPlace_prop)
        self.save_data(self, data_dir, "test/deathPlace/", "test", test_deathplace_prop)
        self.save_data(self, data_dir, "test/award/", "test", test_award_prop)
        self.save_data(self, data_dir, "test/subsidiary/", "test", test_subsidiary_prop)
        self.save_data(self, data_dir, "test/author/", "test", test_author_prop)
        self.save_data(self, data_dir, "test/starring/", "test", test_starring_prop)
        self.save_data(self, data_dir, "test/spouse/", "test", test_spouse_prop)
        self.save_data(self, data_dir, "test/foundationPlace/", "test", test_foundationPlace_prop)
        self.save_data(self, data_dir, "test/birthPlace/", "test", test_birthPlace_prop)
        train_deathplace_prop = []
        train_award_prop = []
        train_subsidiary_prop = []
        train_author_prop = []
        train_starring_prop = []
        train_foundationPlace_prop = []
        train_spouse_prop = []
        train_birthPlace_prop = []
        test_deathplace_prop = []
        test_award_prop = []
        test_subsidiary_prop = []
        test_author_prop = []
        test_starring_prop = []
        test_foundationPlace_prop = []
        test_spouse_prop = []
        test_birthPlace_prop = []


        count = 0
        count1 = 0
        for line in training_data_pred:
            count = count +1
            if str(line[1]).__eq__('<http://dbpedia.org/ontology/deathPlace>'):
                train_deathplace_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/award>'):
                train_award_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/subsidiary>'):
                train_subsidiary_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/author>'):
                train_author_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/starring>'):
                train_starring_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/spouse>'):
                train_spouse_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/foundationPlace>'):
                train_foundationPlace_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/birthPlace>'):
                train_birthPlace_prop.append(line)
            else:
                count = count - 1
                exit(1)



        for line in testing_data_pred:
            count1 = count1 +1
            if str(line[1]).__eq__('<http://dbpedia.org/ontology/deathPlace>'):
                test_deathplace_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/award>'):
                test_award_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/subsidiary>'):
                test_subsidiary_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/author>'):
                test_author_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/starring>'):
                test_starring_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/spouse>'):
                test_spouse_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/foundationPlace>'):
                test_foundationPlace_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/birthPlace>'):
                test_birthPlace_prop.append(line)
            else:
                count1 = count1 - 1
                exit(1)

        self.save_data(self, data_dir,"train/deathPlace/","train_pred",train_deathplace_prop, True)
        self.save_data(self, data_dir,"train/award/","train_pred",train_award_prop, True)
        self.save_data(self, data_dir,"train/subsidiary/","train_pred",train_subsidiary_prop, True)
        self.save_data(self, data_dir,"train/author/","train_pred",train_author_prop, True)
        self.save_data(self, data_dir,"train/starring/","train_pred",train_starring_prop, True)
        self.save_data(self, data_dir,"train/spouse/","train_pred",train_spouse_prop, True)
        self.save_data(self, data_dir,"train/foundationPlace/","train_pred",train_foundationPlace_prop, True)
        self.save_data(self, data_dir,"train/birthPlace/","train_pred",train_birthPlace_prop, True)
        self.save_data(self, data_dir , "test/deathPlace/", "test_pred", test_deathplace_prop, True)
        self.save_data(self, data_dir , "test/award/", "test_pred", test_award_prop, True)
        self.save_data(self, data_dir , "test/subsidiary/", "test_pred", test_subsidiary_prop, True)
        self.save_data(self, data_dir , "test/author/", "test_pred", test_author_prop, True)
        self.save_data(self, data_dir , "test/starring/", "test_pred", test_starring_prop, True)
        self.save_data(self, data_dir , "test/spouse/", "test_pred", test_spouse_prop, True)
        self.save_data(self, data_dir , "test/foundationPlace/", "test_pred", test_foundationPlace_prop, True)
        self.save_data(self, data_dir , "test/birthPlace/", "test_pred", test_birthPlace_prop, True)
        print("properties split completed")


    @staticmethod
    def save_data(self, data_dir, folder, file_name,arr,pred=False):
        with open(data_dir + "properties_split/"+folder + file_name+".txt", "w") as file:
            new_line = ""
            b = False
            if pred==False:
                for idx, r1 in enumerate(arr):
                    if r1[3]==1:
                        b = True
                    else:
                        b = False
                    new_line += r1[0]+"  "+r1[1]+"  "+r1[2] +"  "+ str(b) + "\n"
            else:
                for idx, r1 in enumerate(arr):
                    new_line += r1[0]+"  "+r1[1]+"  "+r1[2] +"  "+ str(r1[3]) + "\n"

            file.write(new_line)
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
                        else:
                            raise ValueError
            else:
                with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                    for datapoint in f:
                        datapoint = datapoint.split()
                        if len(datapoint) == 4:
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












path_dataset_folder = '../dataset/'
se = SeperatePropertiesData(data_dir=path_dataset_folder,multiclass=True)

