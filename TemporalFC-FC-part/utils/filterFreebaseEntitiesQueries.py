from rdflib import Graph, URIRef, OWL

from utils.dataSingleFile import DataSingleFile

countFb = 0


# FIRST STEP
# extracting evedience sentence and generating training and testing triples from the list of facts in factbecnh.
class filterFreebaseEntitiesQueries:

    def __init__(self, data_dir=None,multiclass=True):
        # save all entities and relations from final_output file to new files-> comment these 2 lines if not needed
        # self.store_all_entities_and_relations(self, path_dataset_folder,"final_output")
        # exit(1)
        # ///////////////////////////////////////////////////////////////////////////////

        global countFb
        self.dataset_file= "textResults4.txt"
        if multiclass:
            self.extract_sentence_embeddings_from_factcheck_output_multiclass(self,data_dir)
        # else:
        #     self.extract_sentence_embeddings_from_factcheck_output(self,data_dir)


    # @staticmethod
    # def extract_sentence_embeddings_from_factcheck_output(self,data_dir):
    #     data_train = []
    #     data_test = []
    #     multiclass_neg_exp = True
    #     pred = []
    #     test = False
    #     train = False
    #     correct = False
    #
    #     with open(data_dir+ self.dataset_file, "r") as file1:
    #         for line in file1:
    #             if line.__contains__("factbench/factbench/test/"):
    #                 test = True
    #                 train = False
    #                 if line.__contains__("/wrong/"):
    #                     correct = False
    #                 else:
    #                     correct = True
    #                 continue
    #             if line.__contains__("factbench/factbench/train/"):
    #                 test = False
    #                 train = True
    #                 if line.__contains__("/wrong/"):
    #                     correct = False
    #                 else:
    #                     correct = True
    #
    #                 continue
    #
    #             if line.startswith(' defactoScore'):
    #                 l1 = line.split("defactoScore: ")[1]
    #                 score = l1.split(" setProofSentences")[0]
    #                 x = line.split("subject : ")[1]
    #                 so = x.split(" object : ")
    #                 s = so[0].replace(" ", "_")
    #                 print(s)
    #                 assert s != ""
    #                 o = so[1].split(" predicate ")[0].replace(" ", "_")
    #                 print(o)
    #                 if o == "": o = "DUMMY"
    #                 assert o != ""
    #                 p = so[1].split(" predicate ")[1].replace("\n", "")
    #                 print(p)
    #                 assert p != ""
    #                 print("line:" + line + ":" + score + ":" + str(correct))
    #
    #                 if test == True and train == False:
    #                     print("test")
    #                     data_test.append([s, p, o, correct])
    #
    #                 if test == False and train == True:
    #                     print("train")
    #                     data_train.append([s, p, o, correct])
    #
    #     with open(data_dir+ "train.txt", "w") as prediction_file:
    #         new_line = ""
    #         for idx, (head, relation, tail, lbl) in enumerate(data_train):
    #             new_line += head + "\t" + relation + "\t" + tail + "\t" + str(lbl) + "\n"
    #
    #         prediction_file.write(new_line)
    #
    #     with open(data_dir+"test.txt", "w") as prediction_file:
    #         new_line = ""
    #         for idx, (head, relation, tail, lbl) in enumerate(data_test):
    #             new_line += head + "\t" + relation + "\t" + tail + "\t" + str(lbl) + "\n"
    #
    #         prediction_file.write(new_line)

    @staticmethod
    def extractURI(self, data_dir, sub, pred, obj,final_output, lbl):
        print(data_dir)
        g = Graph()
        g.parse(data_dir)
        if sub.__contains__("%"):
            print("test")
        isFreebase = False
        print(len(g))
        global countFb
        new_triple = []
        # for stmt in g:
        #     pprint.pprint(stmt)

        for s, p, o in g.triples((None, URIRef('http://dbpedia.org/ontology/'+pred), None)):
            if (not (None, None, s) in g):
                print("error")
                print(g.triples(None, None, s))
                exit(1)
            for s1, p1, o1 in  g.triples((None, None, s)):
                if str(s1).__contains__("freebase"):
                    for s2, p2, o2 in g.triples((s1, OWL.sameAs, None)):
                        if str(o2).startswith("http://dbpedia.org/"):
                            # o2 = str(o2).replace("http://dbpedia.org/resource/","")
                            # updated here o2
                            s1 = o2
                            break
                        if str(o2).startswith("http://en.dbpedia.org/"):
                            # o2 = str(o2).replace("http://en.dbpedia.org/resource/", "")
                            # updated here o2
                            s1 = o2
                            break
                # print(s1, p, o)
                if str(s1).__contains__("freebase"):
                    countFb = countFb +1
                    s1 = "http://dbpedia.org/resource/"+ sub.replace(" ","_")
                    isFreebase = True

                s = s1


            for s1, p1, o1 in g.triples((None, None, o)):
                if str(o1).__contains__("freebase"):
                    for s2, p2, o2 in g.triples((o1, OWL.sameAs, None)):
                        if str(o2).startswith("http://dbpedia.org/"):
                            # o2 = str(o2).replace("http://dbpedia.org/resource/", "")
                            # updated here o2
                            o1 = o2
                            break
                        if str(o2).startswith("http://en.dbpedia.org/"):
                            # o2 = str(o2).replace("http://en.dbpedia.org/resource/", "")
                            # updated here o2
                            o1 = o2
                            break
                # print(s, p, o1)
                if str(o1).__contains__("freebase"):
                    countFb = countFb +1
                    isFreebase = True
                    o1 =  "http://dbpedia.org/resource/"+ obj.replace(" ","_")

                o = o1

            #     print(s, p, o)
            # print((s), (p), (o))
            # new_triple = "<"+s+">", "<"+p+">", "<"+o+">" , lbl
            # final_output.remove([new_triple])
            # final_output.append(new_triple)

        print(s, p, o)
        if str(p).__contains__("dbpedia.org/ontology/office"):
            isFreebase = True
        return "<"+s+">", "<"+p+">", "<"+o+">", isFreebase




    @staticmethod
    def extract_sentence_embeddings_from_factcheck_output_multiclass(self, data_dir):
        data_train = []
        data_test = []
        date_data_train = []
        date_data_test = []
        final_output = []
        final_output_test = []
        final_output_train = []
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
        ddr = ""
        counter = 1
        with open(data_dir + "textResult6.txt", "w") as file:
            print("start")
        with open(data_dir+self.dataset_file, "r", encoding="UTF-8") as file1:
            for line in file1:
                if line.__contains__("factbench/factbench/test/"):
                    test = True
                    train = False
                    if line.__contains__("/wrong/"):
                        correct = False
                    else:
                        correct = True
                    neg_data_dir = "true"
                    ddr = line.replace("\n","")
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
                    ddr = line.replace("\n", "")
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
                    if p.__contains__("office"):
                        print("ppp")
                    print(p)
                    assert p != ""
                    print("line:" + line + ":" + score + ":" + str(correct))
                    uri_s,uri_p,uri_o, isFb = self.extractURI(self, ddr, s, p, o, final_output,str(correct))
                    if not isFb:
                        with open(data_dir + "textResult6.txt", "a") as file:
                            file.write(str(counter)+"\n")
                            counter += 1
                            file.write(ddr+"\n")
                            file.write(line)
                            file.write("-=-=-=-=-=-=-==-=-==-=-=-=-==-=-=-=-=-=-=-==-=-=-=-"+"\n")

                    if test == True and train == False:
                        print("test")
                        data_test.append([uri_s, uri_p, uri_o, correct])
                        data_test1.append([uri_s, uri_p, uri_o, score])

                    if test == False and train == True:
                        print("train")
                        data_train.append([uri_s, uri_p, uri_o, correct])
                        data_train1.append([uri_s, uri_p, uri_o, score])

                    if multiclass_neg_exp == True:
                        if test == False and train == True:
                            if neg_data_dir.__contains__("/date/") or neg_data_dir.__contains__("true"):
                                date_data_train.append([uri_s, uri_p, uri_o, correct])
                                date_data_train1.append([uri_s, uri_p, uri_o, score])
                            if neg_data_dir.__contains__("/domain/") or neg_data_dir.__contains__("true"):
                                domain_data_train.append([uri_s, uri_p, uri_o, correct])
                                domain_data_train1.append([uri_s, uri_p, uri_o, score])
                            if neg_data_dir.__contains__("/domainrange/") or neg_data_dir.__contains__("true"):
                                domainrange_data_train.append([uri_s, uri_p, uri_o, correct])
                                domainrange_data_train1.append([uri_s, uri_p, uri_o, score])
                            if neg_data_dir.__contains__("/mix/") or neg_data_dir.__contains__("true"):
                                mix_data_train.append([uri_s, uri_p, uri_o, correct])
                                mix_data_train1.append([uri_s, uri_p, uri_o, score])
                            if neg_data_dir.__contains__("/property/") or neg_data_dir.__contains__("true"):
                                property_data_train.append([uri_s, uri_p, uri_o, correct])
                                property_data_train1.append([uri_s, uri_p, uri_o, score])
                            if neg_data_dir.__contains__("/random/") or neg_data_dir.__contains__("true"):
                                random_data_train.append([uri_s, uri_p, uri_o, correct])
                                random_data_train1.append([uri_s, uri_p, uri_o, score])
                            if neg_data_dir.__contains__("/range/") or neg_data_dir.__contains__("true"):
                                range_data_train.append([uri_s, uri_p, uri_o, correct])
                                range_data_train1.append([uri_s, uri_p, uri_o, score])

                        if test == True and train == False:
                            if neg_data_dir.__contains__("/date/") or neg_data_dir.__contains__("true"):
                                date_data_test.append([uri_s, uri_p, uri_o, correct])
                                date_data_test1.append([uri_s, uri_p, uri_o, score])
                            if neg_data_dir.__contains__("/domain/") or neg_data_dir.__contains__("true"):
                                domain_data_test.append([uri_s, uri_p, uri_o, correct])
                                domain_data_test1.append([uri_s, uri_p, uri_o, score])
                            if neg_data_dir.__contains__("/domainrange/") or neg_data_dir.__contains__("true"):
                                domainrange_data_test.append([uri_s, uri_p, uri_o, correct])
                                domainrange_data_test1.append([uri_s, uri_p, uri_o, score])
                            if neg_data_dir.__contains__("/mix/") or neg_data_dir.__contains__("true"):
                                mix_data_test.append([uri_s, uri_p, uri_o, correct])
                                mix_data_test1.append([uri_s, uri_p, uri_o, score])
                            if neg_data_dir.__contains__("/property/") or neg_data_dir.__contains__("true"):
                                property_data_test.append([uri_s, uri_p, uri_o, correct])
                                property_data_test1.append([uri_s, uri_p, uri_o, score])
                            if neg_data_dir.__contains__("/random/") or neg_data_dir.__contains__("true"):
                                random_data_test.append([uri_s, uri_p, uri_o, correct])
                                random_data_test1.append([uri_s, uri_p, uri_o, score])
                            if neg_data_dir.__contains__("/range/") or neg_data_dir.__contains__("true"):
                                range_data_test.append([uri_s, uri_p, uri_o, correct])
                                range_data_test1.append([uri_s, uri_p, uri_o, score])

# /////////////////////
        print("total_fb"+str(countFb))

        # exit(1)

        # with open(data_dir+"final_output" + ".txt", "r") as file:
        #     for x in file.readlines():
        #        print(x)
# /////////////////////////


        print("total freebase entities:"+str(countFb))

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

    @staticmethod
    def store_all_entities_and_relations(self, data_dir=None, subpath=None):
        print("storing all entities to a file")
        dataset1 = DataSingleFile(data_dir=data_dir, subpath=subpath)
        with open(data_dir+"complete_dataset/"+"all_entities.txt", "w") as data_file:
            new_line = ""
            for idx, e1 in enumerate(dataset1.entities):
                new_line += e1 + "\n"
            data_file.write(new_line)

        with open(data_dir + "complete_dataset/" + "all_relations.txt", "w") as data_file2:
            new_line = ""
            for idx, r1 in enumerate(dataset1.relations):
                new_line += r1 + "\n"
            data_file2.write(new_line)


path_dataset_folder = '../dataset/'
se = filterFreebaseEntitiesQueries(data_dir=path_dataset_folder,multiclass=True)
