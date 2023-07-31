# from torch import hub
from sentence_transformers import SentenceTransformer

from data import Data
import numpy as np
import pandas as pd
from rdflib import Graph, URIRef, OWL
# from sentence_transformers import SentenceTransformer
from select_top_n_sentences import select_top_n_sentences
import zipfile
import logging
logging.basicConfig(level=logging.INFO)
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# extracting evedience sentence and generating embeddings from those evedence sentences and storing them in CSV file format
class SentenceEmbeddings:
    def __init__(self, data_dir=None,multiclass=True):
        self.data_file = "textResult7.txt"
        if multiclass:
            self.extract_sentence_embeddings_from_factcheck_output_multiclass(self, data_dir)
        else:
            self.extract_sentence_embeddings_from_factcheck_output(self,data_dir)



    @staticmethod
    def extract_sentence_embeddings_from_factcheck_output(self,data_dir=None):
        data_train = []
        data_test = []
        pred = []
        test = False
        train = False
        correct = False
        print(data_dir)
        # dataset1 = Data(data_dir=data_dir)
        with open(data_dir+self.data_file, "r") as file1:
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
                    o = so[1].split(" predicate ")[0].replace(" ", "_")
                    if o == "" : o = "DUMMY"
                    assert o != ""
                    p = so[1].split(" predicate ")[1].replace("\n", "")
                    # print("line:" + line+ ":"+ score + ":"+ str(correct))
                    sentences = []
                    if line.__contains__("[ComplexProofs{"):
                        print("line:" + line + ":" + score + ":" + str(correct))
                        for idx, proof in enumerate(line.split("ComplexProofs{")):
                            if idx == 0:
                                continue
                            # print (str(idx) + ":" +proof)
                            p1 = proof.split("}")[0]
                            p2 = p1.split(", proofPhrase=")[1]
                            print(str(idx) + ":" + p2)
                            sentences.append(p2)

                    if test == True and train == False:
                        print("test" + str(correct))
                        data_test.append([s, p, o, correct, sentences])

                    if test == False and train == True:
                        print("train" + str(correct))
                        data_train.append([s, p, o, correct, sentences])

        model = SentenceTransformer('all-mpnet-base-v2')
        embeddings_textual_evedences_train = []
        embeddings_textual_evedences_test = []
        for idx, (s, p, o, c, p2,t1) in enumerate(data_train):
            print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
            # Sentences are encoded by calling model.encode()
            sentence_embeddings = model.encode(p2)

            for s2 in p2:
                print("\n" + s2)
            avg_embedding = np.mean(sentence_embeddings, axis=0)

            if (np.isnan(np.sum(avg_embedding))):
                avg_embedding = np.zeros(768, dtype=int)

            embeddings_textual_evedences_train.append(avg_embedding)

        triple_emb = model.encode(s + " " + p + " " + o)
        for idx, (s, p, o, c, p2,t1) in enumerate(data_test):
            print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
            p3 = p2.copy()
            for idx2, ss in enumerate(p3):
                st1 = model.encode(ss)
                if cosine(triple_emb, st1) < 0.1:
                    p2.remove(ss)
                else:
                    print(cosine(triple_emb, st1))

            # Sentences are encoded by calling model.encode()
            sentence_embeddings = model.encode(p2)

            for s2 in p2:
                print("\n" + s2)
            avg_embedding = np.mean(sentence_embeddings, axis=0)

            print(avg_embedding)
            if (np.isnan(np.sum(avg_embedding))):
                avg_embedding = np.zeros(768, dtype=int)
                print(avg_embedding)
                # exit(1)

            embeddings_textual_evedences_test.append(avg_embedding)



        X = np.array(embeddings_textual_evedences_train)
        print(X.shape)
        X=pd.DataFrame(X)
        compression_opts = dict(method='zip',archive_name='trainSE.csv')
        X.to_csv(data_dir+'trainSE.zip', index=False,compression=compression_opts)
        with zipfile.ZipFile(data_dir+'trainSE.zip', 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        Y = np.array(embeddings_textual_evedences_test)
        print(Y.shape)
        Y=pd.DataFrame(Y)
        compression_opts1 = dict(method='zip',archive_name='testSE.csv')
        Y.to_csv(data_dir+'testSE.zip', index=False,compression=compression_opts1)
        with zipfile.ZipFile(data_dir+'testSE.zip', 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    @staticmethod
    def entityToNLrepresentation(self, predicate):
        p = predicate
        if p == "birthPlace":
            p = "born in"
        if p == "deathPlace":
            p = "died in"
        if p == "foundationPlace":
            p = "founded in"
        if p == "starring":
            p = "starring in"
        if p == "award":
            p = "awarded with"
        if p == "subsidiary":
            p = "subsidiary of"
        if p == "author":
            p = "author of"
        if p == "spouse":
            p = "spouse of"
        if p == "office":
            p = "office"
        if p == "team":
            p = "team"
        return p

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
        # keys = dictionaries[0].keys()
        # a_file = open("output.csv", "w")
        # dict_writer = csv.DictWriter(a_file, keys)
        # dict_writer.writeheader()
        # dict_writer.writerows(dictionaries)
        # a_file.close()
    @staticmethod
    def getSentenceEmbeddings(self,data_dir, data, model, true_statements_embeddings, type = 'default', n =3):
        embeddings_textual_evedences = []
        website_ids = set()
        for idx, (s, p, o, c, p2, t1) in enumerate(data):
            # p = self.entityToNLrepresentation(self, p)
            path = data_dir + "data/test/" + type.replace("_test", "") + "/"
            print('triple->'+s+' '+p+' '+o)
            # triple_emb = model.encode(s + " " + p + " " + o.replace("_", " "))
            # print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
            p3 = p2.copy()
            temp = []
            temp2 = []
            for tt in enumerate(p3):
                temp.append(tt[1].split(",website=")[0])
                temp2.append((tt[1].split(",website=")[1]).split("/")[-1])

            # # Sentences are encoded by calling model.encode()
            for item in temp2:
                website_ids.add(item)

            sentence_embeddings = dict()
            if (s+' '+p+' '+o) not in true_statements_embeddings.keys():
                temp22, temp33 = select_top_n_sentences(temp2, n, temp,type)
                sentence_embeddings[s+' '+p+' '+o]  = model.encode(temp33)
                true_statements_embeddings[s+' '+p+' '+o] = sentence_embeddings[s+' '+p+' '+o]
            else:
                sentence_embeddings[s+' '+p+' '+o]  = true_statements_embeddings[s+' '+p+' '+o]


            if (np.size(sentence_embeddings[s+' '+p+' '+o] ) == 0):
                sentence_embeddings[s+' '+p+' '+o]  = np.zeros((n, 768), dtype=int)

            if (np.size(sentence_embeddings[s+' '+p+' '+o] ) == 768*(n-2)):
                sentence_embeddings[s+' '+p+' '+o]  = np.append(sentence_embeddings[s+' '+p+' '+o] ,(np.zeros((n-1, 768), dtype=int)), axis=0)

            if (np.size(sentence_embeddings[s+' '+p+' '+o] ) == 768*(n-1)):
                sentence_embeddings[s+' '+p+' '+o]  = np.append(sentence_embeddings[s+' '+p+' '+o] ,(np.zeros((n-2, 768), dtype=int)), axis=0)

            embeddings_textual_evedences.append(sentence_embeddings)
        # with open('/home/umair/Documents/pythonProjects/HybridFactChecking/dataset/pg_ranks/' + 'all_websites_ids_'+type+'.txt',"w") as f:
        #     for item in list(website_ids):
        #         f.write("%s\n" % item)
        path = ""
        if type.__contains__("test"):
            path = data_dir + "data/test/" + type.replace("_test","") + "/"
            self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences, "test", n)
        else:
            path = data_dir + "data/train/" + type + "/"
            self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences, "train", n)
        return embeddings_textual_evedences, true_statements_embeddings


    @staticmethod
    def extractURI(self, data_dir, sub, pred, obj):
        if obj == 'Quincy,_Massachusetts' and pred == 'spouse':
            print("test")
        print(data_dir)
        g = Graph()
        g.parse(data_dir)
        if sub.__contains__("%"):
            print("test")
        global countFb
        for s, p, o in g.triples((None, URIRef('http://dbpedia.org/ontology/'+pred), None)):
            if (not (None, None, s) in g):
                print("error")
                print(g.triples(None, None, s))
                exit(1)
            for s1, p1, o1 in  g.triples((None, None, s)):
                if str(s1).__contains__("freebase"):
                    for s2, p2, o2 in g.triples((s1, OWL.sameAs, None)):
                        if str(o2).startswith("http://dbpedia.org/"):
                            s1 = o2
                            break
                        if str(o2).startswith("http://en.dbpedia.org/"):
                            s1 = o2
                            break
                if str(s1).__contains__("freebase"):
                    countFb = countFb +1
                    s1 = "http://dbpedia.org/resource/"+ sub.replace(" ","_")
                s = s1
            for s1, p1, o1 in g.triples((None, None, o)):
                if str(o1).__contains__("freebase"):
                    for s2, p2, o2 in g.triples((o1, OWL.sameAs, None)):
                        if str(o2).startswith("http://dbpedia.org/"):
                            o1 = o2
                            break
                        if str(o2).startswith("http://en.dbpedia.org/"):
                            o1 = o2
                            break
                if str(o1).__contains__("freebase"):
                    countFb = countFb +1
                    o1 =  "http://dbpedia.org/resource/"+ obj.replace(" ","_")
                o = o1
        try:
            print(s, p, o)
        except:
            print("s"+s)
        return "<"+s+">", "<"+p+">", "<"+o+">"

    @staticmethod
    def extract_sentence_embeddings_from_factcheck_output_multiclass(self,data_dir=None):
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
        multiclass_neg_exp = True

        pred = []
        test = False
        train = False
        correct = False
        print(data_dir)
        # dataset1 = Data(data_dir=data_dir)
        with open(data_dir+self.data_file, "r") as file1:
            for line in file1:
                if line.__contains__("factbench/factbench/test/"):
                    test = True
                    train = False
                    if line.__contains__("/wrong/"):
                        correct = False
                    else:
                        correct = True

                    neg_data_dir = "true"
                    ddr = line.replace("\n", "")
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
                    o = so[1].split(" predicate ")[0].replace(" ", "_")
                    if o == "" : o = "DUMMY"
                    assert o != ""
                    p = so[1].split(" predicate ")[1].replace("\n", "")
                    # print("line:" + line+ ":"+ score + ":"+ str(correct))
                    s, p, o = self.extractURI(self, data_dir=ddr, sub=s, pred=p, obj=o)
                    sentences = []
                    websites = []
                    trustworthiness = []

                    if line.__contains__("[ComplexProofs{"):
                        print("line:" + line + ":" + score + ":" + str(correct))
                        for idx, proof in enumerate(line.split("ComplexProofs{")):
                            if idx == 0:
                                continue
                            # print (str(idx) + ":" +proof)
                            p1 = proof.split("}")[0]
                            website = proof.split("website='")[1].split("', proofPhrase")[0].replace(" ","_")
                            p2 = p1.split(", proofPhrase='")[1]
                            p3 = p2.split("', trustworthinessScore='")
                            print(str(idx) + ":" + p3[0] + "tr: " +p3[1][:-1])
                            sentences.append(p3[0] + ",website="+website)
                            websites.append(website)
                            trustworthiness.append(p3[1][:-1])

                    if test == True and train == False:
                        print("test" + str(correct))
                        data_test.append([s, p, o, correct, sentences,trustworthiness])

                    if test == False and train == True:
                        print("train" + str(correct))
                        data_train.append([s, p, o, correct, sentences,trustworthiness])

                    if multiclass_neg_exp == True:
                        if test == False and train == True:
                            if neg_data_dir.__contains__("/date/") or neg_data_dir.__contains__("true"):
                                date_data_train.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/domain/") or neg_data_dir.__contains__("true"):
                                domain_data_train.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/domainrange/") or neg_data_dir.__contains__("true"):
                                domainrange_data_train.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/mix/") or neg_data_dir.__contains__("true"):
                                mix_data_train.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/property/") or neg_data_dir.__contains__("true"):
                                property_data_train.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/random/") or neg_data_dir.__contains__("true"):
                                random_data_train.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/range/") or neg_data_dir.__contains__("true"):
                                range_data_train.append([s, p, o, correct, sentences,trustworthiness])

                        if test == True and train == False:
                            if neg_data_dir.__contains__("/date/") or neg_data_dir.__contains__("true"):
                                date_data_test.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/domain/") or neg_data_dir.__contains__("true"):
                                domain_data_test.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/domainrange/") or neg_data_dir.__contains__("true"):
                                domainrange_data_test.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/mix/") or neg_data_dir.__contains__("true"):
                                mix_data_test.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/property/") or neg_data_dir.__contains__("true"):
                                property_data_test.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/random/") or neg_data_dir.__contains__("true"):
                                random_data_test.append([s, p, o, correct, sentences,trustworthiness])
                            if neg_data_dir.__contains__("/range/") or neg_data_dir.__contains__("true"):
                                range_data_test.append([s, p, o, correct, sentences,trustworthiness])


        model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
        # model = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/2")

        true_statements_embeddings = {}
        n = 3
        # ///////////////////////////////////////////////////////////////////////////////train
        embeddings_textual_evedences_train_date, true_statements_embeddings = self.getSentenceEmbeddings(self,data_dir,date_data_train,model,true_statements_embeddings, 'date',n)

        embeddings_textual_evedences_train_domain, true_statements_embeddings = self.getSentenceEmbeddings(self, data_dir,domain_data_train, model,true_statements_embeddings, 'domain',n)

        embeddings_textual_evedences_train_domainrange, true_statements_embeddings = self.getSentenceEmbeddings(self,data_dir, domainrange_data_train, model,true_statements_embeddings, 'domainrange',n)

        embeddings_textual_evedences_train_mix, true_statements_embeddings = self.getSentenceEmbeddings(self,data_dir, mix_data_train, model,true_statements_embeddings, 'mix',n)

        embeddings_textual_evedences_train_property, true_statements_embeddings = self.getSentenceEmbeddings(self,data_dir, property_data_train, model,true_statements_embeddings, 'property',n)

        embeddings_textual_evedences_train_random,true_statements_embeddings = self.getSentenceEmbeddings(self,data_dir, random_data_train, model,true_statements_embeddings, 'random',n)

        embeddings_textual_evedences_train_range,true_statements_embeddings = self.getSentenceEmbeddings(self,data_dir, range_data_train, model,true_statements_embeddings, 'range',n)

        true_statements_embeddings_test = {}
        embeddings_textual_evedences_test_date, true_statements_embeddings_test = self.getSentenceEmbeddings(self,data_dir,date_data_test,model, true_statements_embeddings_test, 'date_test',n)
        embeddings_textual_evedences_test_domain,true_statements_embeddings_test = self.getSentenceEmbeddings(self,data_dir, domain_data_test, model,true_statements_embeddings_test, 'domain_test',n)
        embeddings_textual_evedences_test_domainrange, true_statements_embeddings_test = self.getSentenceEmbeddings(self,data_dir, domainrange_data_test, model, true_statements_embeddings_test, 'domainrange_test',n)
        embeddings_textual_evedences_test_mix,true_statements_embeddings_test = self.getSentenceEmbeddings(self,data_dir, mix_data_test, model,true_statements_embeddings_test, 'mix_test',n)
        embeddings_textual_evedences_test_property,true_statements_embeddings_test = self.getSentenceEmbeddings(self, data_dir,property_data_test, model,true_statements_embeddings_test, 'property_test',n)
        embeddings_textual_evedences_test_random,true_statements_embeddings_test = self.getSentenceEmbeddings(self,data_dir, random_data_test, model,true_statements_embeddings_test, 'random_test',n)
        embeddings_textual_evedences_test_range,true_statements_embeddings_test = self.getSentenceEmbeddings(self,data_dir, range_data_test, model,true_statements_embeddings_test, 'range_test',n)

        @staticmethod
        def extract_sentence_embeddings_from_factcheck_output_multiproperties(self, data_dir=None):
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

            multiclass_neg_exp = True

            pred = []
            test = False
            train = False
            correct = False
            print(data_dir)
            # dataset1 = Data(data_dir=data_dir)
            with open(data_dir + self.data_file, "r") as file1:
                for line in file1:
                    if line.__contains__("factbench/factbench/test/"):
                        test = True
                        train = False
                        if line.__contains__("/wrong/"):
                            correct = False
                        else:
                            correct = True

                        neg_data_dir = "true"
                        ddr = line.replace("\n", "")
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
                        o = so[1].split(" predicate ")[0].replace(" ", "_")
                        if o == "": o = "DUMMY"
                        assert o != ""
                        p = so[1].split(" predicate ")[1].replace("\n", "")
                        # print("line:" + line+ ":"+ score + ":"+ str(correct))
                        s, p, o = self.extractURI(self, data_dir=ddr, sub=s, pred=p, obj=o)
                        sentences = []
                        websites = []
                        trustworthiness = []

                        if line.__contains__("[ComplexProofs{"):
                            print("line:" + line + ":" + score + ":" + str(correct))
                            for idx, proof in enumerate(line.split("ComplexProofs{")):
                                if idx == 0:
                                    continue
                                # print (str(idx) + ":" +proof)
                                p1 = proof.split("}")[0]
                                website = proof.split("website='")[1].split("', proofPhrase")[0].replace(" ", "_")
                                p2 = p1.split(", proofPhrase='")[1]
                                p3 = p2.split("', trustworthinessScore='")
                                print(str(idx) + ":" + p3[0] + "tr: " + p3[1][:-1])
                                sentences.append(p3[0] + ",website=" + website)
                                websites.append(website)
                                trustworthiness.append(p3[1][:-1])

                        if test == True and train == False:
                            print("test" + str(correct))
                            data_test.append([s, p, o, correct, sentences, trustworthiness])

                        if test == False and train == True:
                            print("train" + str(correct))
                            data_train.append([s, p, o, correct, sentences, trustworthiness])

                        if multiclass_neg_exp == True:
                            if test == False and train == True:
                                if neg_data_dir.__contains__("/date/") or neg_data_dir.__contains__("true"):
                                    date_data_train.append([s, p, o, correct, sentences, trustworthiness])
                                if neg_data_dir.__contains__("/domain/") or neg_data_dir.__contains__("true"):
                                    domain_data_train.append([s, p, o, correct, sentences, trustworthiness])
                                if neg_data_dir.__contains__("/domainrange/") or neg_data_dir.__contains__("true"):
                                    domainrange_data_train.append([s, p, o, correct, sentences, trustworthiness])
                                if neg_data_dir.__contains__("/mix/") or neg_data_dir.__contains__("true"):
                                    mix_data_train.append([s, p, o, correct, sentences, trustworthiness])
                                if neg_data_dir.__contains__("/property/") or neg_data_dir.__contains__("true"):
                                    property_data_train.append([s, p, o, correct, sentences, trustworthiness])
                                if neg_data_dir.__contains__("/random/") or neg_data_dir.__contains__("true"):
                                    random_data_train.append([s, p, o, correct, sentences, trustworthiness])
                                if neg_data_dir.__contains__("/range/") or neg_data_dir.__contains__("true"):
                                    range_data_train.append([s, p, o, correct, sentences, trustworthiness])

                            if test == True and train == False:
                                if neg_data_dir.__contains__("/date/") or neg_data_dir.__contains__("true"):
                                    date_data_test.append([s, p, o, correct, sentences, trustworthiness])
                                if neg_data_dir.__contains__("/domain/") or neg_data_dir.__contains__("true"):
                                    domain_data_test.append([s, p, o, correct, sentences, trustworthiness])
                                if neg_data_dir.__contains__("/domainrange/") or neg_data_dir.__contains__("true"):
                                    domainrange_data_test.append([s, p, o, correct, sentences, trustworthiness])
                                if neg_data_dir.__contains__("/mix/") or neg_data_dir.__contains__("true"):
                                    mix_data_test.append([s, p, o, correct, sentences, trustworthiness])
                                if neg_data_dir.__contains__("/property/") or neg_data_dir.__contains__("true"):
                                    property_data_test.append([s, p, o, correct, sentences, trustworthiness])
                                if neg_data_dir.__contains__("/random/") or neg_data_dir.__contains__("true"):
                                    random_data_test.append([s, p, o, correct, sentences, trustworthiness])
                                if neg_data_dir.__contains__("/range/") or neg_data_dir.__contains__("true"):
                                    range_data_test.append([s, p, o, correct, sentences, trustworthiness])

            model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
            # model = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/2")

            true_statements_embeddings = {}
            n = 3
            # ///////////////////////////////////////////////////////////////////////////////train
            embeddings_textual_evedences_train_date, true_statements_embeddings = self.getSentenceEmbeddings(self,
                                                                                                             data_dir,
                                                                                                             date_data_train,
                                                                                                             model,
                                                                                                             true_statements_embeddings,
                                                                                                             'date', n)

            embeddings_textual_evedences_train_domain, true_statements_embeddings = self.getSentenceEmbeddings(self,
                                                                                                               data_dir,
                                                                                                               domain_data_train,
                                                                                                               model,
                                                                                                               true_statements_embeddings,
                                                                                                               'domain',
                                                                                                               n)

            embeddings_textual_evedences_train_domainrange, true_statements_embeddings = self.getSentenceEmbeddings(
                self, data_dir, domainrange_data_train, model, true_statements_embeddings, 'domainrange', n)

            embeddings_textual_evedences_train_mix, true_statements_embeddings = self.getSentenceEmbeddings(self,
                                                                                                            data_dir,
                                                                                                            mix_data_train,
                                                                                                            model,
                                                                                                            true_statements_embeddings,
                                                                                                            'mix', n)

            embeddings_textual_evedences_train_property, true_statements_embeddings = self.getSentenceEmbeddings(self,
                                                                                                                 data_dir,
                                                                                                                 property_data_train,
                                                                                                                 model,
                                                                                                                 true_statements_embeddings,
                                                                                                                 'property',
                                                                                                                 n)

            embeddings_textual_evedences_train_random, true_statements_embeddings = self.getSentenceEmbeddings(self,
                                                                                                               data_dir,
                                                                                                               random_data_train,
                                                                                                               model,
                                                                                                               true_statements_embeddings,
                                                                                                               'random',
                                                                                                               n)

            embeddings_textual_evedences_train_range, true_statements_embeddings = self.getSentenceEmbeddings(self,
                                                                                                              data_dir,
                                                                                                              range_data_train,
                                                                                                              model,
                                                                                                              true_statements_embeddings,
                                                                                                              'range',
                                                                                                              n)

            true_statements_embeddings_test = {}
            embeddings_textual_evedences_test_date, true_statements_embeddings_test = self.getSentenceEmbeddings(self,
                                                                                                                 data_dir,
                                                                                                                 date_data_test,
                                                                                                                 model,
                                                                                                                 true_statements_embeddings_test,
                                                                                                                 'date_test',
                                                                                                                 n)
            embeddings_textual_evedences_test_domain, true_statements_embeddings_test = self.getSentenceEmbeddings(self,
                                                                                                                   data_dir,
                                                                                                                   domain_data_test,
                                                                                                                   model,
                                                                                                                   true_statements_embeddings_test,
                                                                                                                   'domain_test',
                                                                                                                   n)
            embeddings_textual_evedences_test_domainrange, true_statements_embeddings_test = self.getSentenceEmbeddings(
                self, data_dir, domainrange_data_test, model, true_statements_embeddings_test, 'domainrange_test', n)
            embeddings_textual_evedences_test_mix, true_statements_embeddings_test = self.getSentenceEmbeddings(self,
                                                                                                                data_dir,
                                                                                                                mix_data_test,
                                                                                                                model,
                                                                                                                true_statements_embeddings_test,
                                                                                                                'mix_test',
                                                                                                                n)
            embeddings_textual_evedences_test_property, true_statements_embeddings_test = self.getSentenceEmbeddings(
                self, data_dir, property_data_test, model, true_statements_embeddings_test, 'property_test', n)
            embeddings_textual_evedences_test_random, true_statements_embeddings_test = self.getSentenceEmbeddings(self,
                                                                                                                   data_dir,
                                                                                                                   random_data_test,
                                                                                                                   model,
                                                                                                                   true_statements_embeddings_test,
                                                                                                                   'random_test',
                                                                                                                   n)
            embeddings_textual_evedences_test_range, true_statements_embeddings_test = self.getSentenceEmbeddings(self,
                                                                                                                  data_dir,
                                                                                                                  range_data_test,
                                                                                                                  model,
                                                                                                                  true_statements_embeddings_test,
                                                                                                                  'range_test',
                                                                                                                  n)

    # for idx, (s, p, o, c, p2,t1) in enumerate(date_data_test):
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #
        #     # Sentences are encoded by calling model.encode()
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #
        #     embeddings_textual_evedences_test_date.append(avg_embedding)
        #
        #
        # for idx, (s, p, o, c, p2,t1) in enumerate(domain_data_test):
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #
        #     # Sentences are encoded by calling model.encode()
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #
        #     embeddings_textual_evedences_test_domain.append(avg_embedding)
        #
        #
        # for idx, (s, p, o, c, p2,t1) in enumerate(domainrange_data_test):
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #
        #     # Sentences are encoded by calling model.encode()
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #
        #     embeddings_textual_evedences_test_domainrange.append(avg_embedding)
        #
        # for idx, (s, p, o, c, p2,t1) in enumerate(mix_data_test):
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #     # Sentences are encoded by calling model.encode()
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #
        #     embeddings_textual_evedences_test_mix.append(avg_embedding)
        #
        # for idx, (s, p, o, c, p2,t1) in enumerate(property_data_test):
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #     # Sentences are encoded by calling model.encode()
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #
        #     embeddings_textual_evedences_test_property.append(avg_embedding)
        #
        # for idx, (s, p, o, c, p2,t1) in enumerate(random_data_test):
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #
        #     # Sentences are encoded by calling model.encode()
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #
        #     embeddings_textual_evedences_test_random.append(avg_embedding)
        #
        # for idx, (s, p, o, c, p2,t1) in enumerate(range_data_test):
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #
        #     # Sentences are encoded by calling model.encode()
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #
        #     embeddings_textual_evedences_test_range.append(avg_embedding)

        # /////////////////////////////////////////////////////////////////////////////// normal
        # embeddings_textual_evedences_train = self.getSentenceEmbeddings(self, data_train, model)
        # embeddings_textual_evedences_test = self.getSentenceEmbeddings(self, data_test, model)
        # for idx, (s, p, o, c, p2,t1) in enumerate(data_train):
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #
        #     # Sentences are encoded by calling model.encode()
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #
        #     embeddings_textual_evedences_train.append(avg_embedding)
        #
        #
        #
        #
        # for idx, (s, p, o, c, p2,t1) in enumerate(data_test):
        #     print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
        #     p3 = p2.copy()
        #     for idx2, ss in enumerate(p3):
        #         st1 = model.encode(ss)
        #         if cosine(triple_emb, st1) < 0.1:
        #             p2.remove(ss)
        #         else:
        #             print(cosine(triple_emb, st1))
        #
        #     # Sentences are encoded by calling model.encode()
        #     sentence_embeddings = model.encode(p2)
        #
        #     for s2 in p2:
        #         print("\n" + s2)
        #     avg_embedding = np.mean(sentence_embeddings, axis=0)
        #
        #     print(avg_embedding)
        #     if (np.isnan(np.sum(avg_embedding))):
        #         avg_embedding = np.zeros(768, dtype=int)
        #         print(avg_embedding)
        #         # exit(1)
        #
        #     embeddings_textual_evedences_test.append(avg_embedding)

        # /////////////////////////////////////////////////////////////////////////////// saving part
        # exit(1)


        # path = data_dir+ "data/train/date/"
        # X = np.array(embeddings_textual_evedences_train_date)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='trainSE.csv')
        # X.to_csv(path + 'trainSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path + 'trainSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/train/domain/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_train_domain, "train")

        # X = np.array(embeddings_textual_evedences_train_domain)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='trainSE.csv')
        # X.to_csv(path + 'trainSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path +'trainSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/train/domainrange/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_train_domainrange, "train")

        # X = np.array(embeddings_textual_evedences_train_domainrange)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='trainSE.csv')
        # X.to_csv(path + 'trainSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path+'trainSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/train/mix/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_train_mix, "train")

        # X = np.array(embeddings_textual_evedences_train_mix)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='trainSE.csv')
        # X.to_csv(path + 'trainSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path +'trainSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/train/property/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_train_property, "train")

        # X = np.array(embeddings_textual_evedences_train_property)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='trainSE.csv')
        # X.to_csv(path + 'trainSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path +'trainSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/train/random/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_train_random, "train")

        # X = np.array(embeddings_textual_evedences_train_random)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='trainSE.csv')
        # X.to_csv(path + 'trainSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path + 'trainSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/train/range/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_train_range, "train")

        # X = np.array(embeddings_textual_evedences_train_range)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='trainSE.csv')
        # X.to_csv(path + 'trainSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path +'trainSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)


        # /////

        # path = data_dir + "data/test/date/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_test_date, "test")
        #
        # X = np.array(embeddings_textual_evedences_test_date)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='testSE.csv')
        # X.to_csv(path + 'testSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path +'testSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/test/domain/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_test_domain, "test")
        #
        # X = np.array(embeddings_textual_evedences_test_domain)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='testSE.csv')
        # X.to_csv(path + 'testSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path +'testSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/test/domainrange/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_test_domainrange, "test")
        #
        # X = np.array(embeddings_textual_evedences_test_domainrange)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='testSE.csv')
        # X.to_csv(path + 'testSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path + 'testSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/test/mix/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_test_mix, "test")
        #
        # X = np.array(embeddings_textual_evedences_test_mix)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='testSE.csv')
        # X.to_csv(path + 'testSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path+'testSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/test/property/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_test_property, "test")
        #
        # X = np.array(embeddings_textual_evedences_test_property)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='testSE.csv')
        # X.to_csv(path + 'testSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path+'testSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/test/random/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_test_random, "test")
        #
        # X = np.array(embeddings_textual_evedences_test_random)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='testSE.csv')
        # X.to_csv(path + 'testSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path+'testSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)

        # path = data_dir + "data/test/range/"
        # self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences_test_range, "test")
        #
        # X = np.array(embeddings_textual_evedences_test_range)
        # print(X.shape)
        # X = pd.DataFrame(X)
        # compression_opts = dict(method='zip', archive_name='testSE.csv')
        # X.to_csv(path + 'testSE.zip', index=False, compression=compression_opts)
        # with zipfile.ZipFile(path+'testSE.zip', 'r') as zip_ref:
        #     zip_ref.extractall(path)
        # /////////////////////////////////////////////////////////////////////////normal


        # X = np.array(embeddings_textual_evedences_train)
        # print(X.shape)
        # X=pd.DataFrame(X)
        # compression_opts = dict(method='zip',archive_name='trainSE.csv')
        # X.to_csv(data_dir+'trainSE.zip', index=False,compression=compression_opts)
        #
        # Y = np.array(embeddings_textual_evedences_test)
        # print(Y.shape)
        # Y=pd.DataFrame(Y)
        # compression_opts1 = dict(method='zip',archive_name='testSE.csv')
        # Y.to_csv(data_dir+'testSE.zip', index=False,compression=compression_opts1)






path_dataset_folder = './dataset/'
se = SentenceEmbeddings(data_dir=path_dataset_folder,multiclass=True)
