# from torch import hub
from sentence_transformers import SentenceTransformer
from urllib.parse import quote, unquote

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
    def __init__(self, data_dir=None,multiclass=True, multiprops = False, dataset_file = "", bpdp = False, copaal = False):
        self.data_file = dataset_file
        # to save website ids first for pagerank scores -> make save_website_ids - True
        save_website_ids = False
        if save_website_ids == True:
            self.extract_sentence_embeddings_from_factcheck_output_bpdp(self, data_dir, get_website_ids=save_website_ids)
            print("Now please generate pagerank scores of the file (by running script) created in page_rank folder in dataset")
            exit()

        if bpdp:
            self.extract_sentence_embeddings_from_factcheck_output_bpdp(self, data_dir)
        elif multiclass:
            self.extract_sentence_embeddings_from_factcheck_output_multiclass(self, data_dir, copaal)
        elif multiprops:
            self.extract_sentence_embeddings_from_factcheck_output_multiproperties(self, data_dir)
        else:
            self.extract_sentence_embeddings_from_factcheck_output(self,data_dir)

    @staticmethod
    def extract_sentence_embeddings_from_factcheck_output_bpdp(self, data_dir=None, get_website_ids=False):
        data_train = []
        data_test = []
        pred = []
        test = False
        train = False
        correct = False
        print(data_dir)
        # dataset1 = Data(data_dir=data_dir)
        with open(data_dir + self.data_file, "r") as file1:
            for line in file1:
                if line.__contains__("/datasets/BPDP_Dataset/Test/"):
                    test = True
                    train = False
                    if line.__contains__("/False/"):
                        correct = False
                    else:
                        correct = True
                    ddr = line.replace("\n", "")
                    continue
                if line.__contains__("/datasets/BPDP_Dataset/Train/"):
                    test = False
                    train = True
                    if line.__contains__("/False/"):
                        correct = False
                    else:
                        correct = True
                    # neg_data_dir = "true"
                    ddr = line.replace("\n", "")
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
                    s, p, o = self.extractURI(self, data_dir=ddr, sub=s, pred=p, obj=o)
                    s, p, o = unquote(s), unquote(p), unquote(o)
                    sentences = []
                    websites = []
                    trustworthiness = []
                    # print("line:" + line+ ":"+ score + ":"+ str(correct))
                    # sentences = []
                    if line.__contains__("[ComplexProofs{"):
                        print("line:" + line + ":" + score + ":" + str(correct))
                        for idx, proof in enumerate(line.split("ComplexProofs{")):
                            if idx == 0:
                                continue
                            # print (str(idx) + ":" +proof)
                            p1 = proof.split("}")[0]
                            website = proof.split("website='")[1].split("', proofPhrase")[0].replace(" ", "_")
                            p2 = p1.split(", proofPhrase=")[1]
                            p3 = p2.split("', trustworthinessScore='")
                            print(str(idx) + ":" + p2)
                            sentences.append(p3[0] + ",website=" + website)
                            websites.append(website)
                            trustworthiness.append(p3[1][:-1])
                    # skip awards relation
                    # if p.__contains__("/ontology/award"):
                    #     continue
                    if test == True and train == False:
                        print("test" + str(correct))
                        data_test.append([s, p, o, correct, sentences, trustworthiness])

                    if test == False and train == True:
                        print("train" + str(correct))
                        data_train.append([s, p, o, correct, sentences, trustworthiness])

        model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
        true_statements_embeddings = {}
        n = 3
        if get_website_ids == True:
            with open('../dataset/pg_ranks/' + 'all_websites_ids_bpdp_complete.txt',"w") as f:
                f.write("")
        embeddings_textual_evedences_train_date, true_statements_embeddings = self.getSentenceEmbeddings(self, data_dir,
                                                                                                         data_train,
                                                                                                         model,
                                                                                                         true_statements_embeddings,
                                                                                                         'bpdp_complete', n,
                                                                                                         False, bpdp=True, web_ids=get_website_ids)
        true_statements_embeddings_test = {}
        embeddings_textual_evedences_test_date, true_statements_embeddings_test = self.getSentenceEmbeddings(self,
                                                                                                             data_dir,
                                                                                                             data_test,
                                                                                                             model,
                                                                                                             true_statements_embeddings_test,
                                                                                                             'bpdp_complete_test',
                                                                                                             n, False, bpdp=True, web_ids=get_website_ids)

        exit(1)
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
                    ddr = line.replace("\n", "")
                    continue
                if line.__contains__("factbench/factbench/train/"):
                    test = False
                    train = True
                    if line.__contains__("/wrong/"):
                        correct = False
                    else:
                        correct = True
                    # neg_data_dir = "true"
                    ddr = line.replace("\n", "")
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
                    s, p, o = self.extractURI(self, data_dir=ddr, sub=s, pred=p, obj=o)
                    s, p, o = unquote(s), unquote(p), unquote(o)
                    sentences = []
                    websites = []
                    trustworthiness = []
                    # print("line:" + line+ ":"+ score + ":"+ str(correct))
                    # sentences = []
                    if line.__contains__("[ComplexProofs{"):
                        print("line:" + line + ":" + score + ":" + str(correct))
                        for idx, proof in enumerate(line.split("ComplexProofs{")):
                            if idx == 0:
                                continue
                            # print (str(idx) + ":" +proof)
                            p1 = proof.split("}")[0]
                            website = proof.split("website='")[1].split("', proofPhrase")[0].replace(" ", "_")
                            p2 = p1.split(", proofPhrase=")[1]
                            p3 = p2.split("', trustworthinessScore='")
                            print(str(idx) + ":" + p2)
                            sentences.append(p3[0] + ",website=" + website)
                            websites.append(website)
                            trustworthiness.append(p3[1][:-1])
                    # skip awards relation
                    # if p.__contains__("/ontology/award"):
                    #     continue
                    if test == True and train == False:
                        print("test" + str(correct))
                        data_test.append([s, p, o, correct, sentences,trustworthiness])

                    if test == False and train == True:
                        print("train" + str(correct))
                        data_train.append([s, p, o, correct, sentences,trustworthiness])

        model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
        true_statements_embeddings = {}
        n = 3
        embeddings_textual_evedences_train_date, true_statements_embeddings = self.getSentenceEmbeddings(self, data_dir,
                                                                                                         data_train,
                                                                                                         model,
                                                                                                         true_statements_embeddings,
                                                                                                         'complete', n, True)
        true_statements_embeddings_test = {}
        embeddings_textual_evedences_test_date, true_statements_embeddings_test = self.getSentenceEmbeddings(self,
                                                                                                             data_dir,
                                                                                                             data_test,
                                                                                                             model,
                                                                                                             true_statements_embeddings_test,
                                                                                                             'complete_test',
                                                                                                             n,  True)

        exit(1)
    @staticmethod
    def save_website_ids(self, data= None, typ = None, nnn = False):
        if nnn:
            with open('/home/umair/Documents/pythonProjects/HybridFactChecking/dataset/pg_ranks/ids_'+typ+'_all_entities.txt',"w") as f:
                f.write("")
                print("start")
        if data!= None:
            with open('/home/umair/Documents/pythonProjects/HybridFactChecking/dataset/pg_ranks/ids_'+typ+'_all_entities.txt',"a") as f:
                for idx, (s, p, o, c, p2, t1) in enumerate(data):
                    p3 = p2.copy()
                    temp = []
                    temp2 = []
                    for tt in enumerate(p3):
                        temp.append(tt[1].split(",website=")[0])
                        temp2.append((tt[1].split(",website=")[1]).split("/")[-1])
                    # # Sentences are encoded by calling model.encode()
                    for item in temp2:
                        f.writelines("%s\n" % item)

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


    @staticmethod
    def get_properties_split(self, data_test, data_dir, model, typ, train= False):
        deathplace_prop = []
        award_prop = []
        subsidiary_prop = []
        author_prop = []
        starring_prop = []
        foundationPlace_prop = []
        spouse_prop = []
        birthPlace_prop = []
        n = 3
        true_statements_embeddings = {}
        count = 0
        for line in data_test:
            count = count +1
            if str(line[1]).__eq__('<http://dbpedia.org/ontology/deathPlace>'):
                deathplace_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/award>'):
                award_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/subsidiary>'):
                subsidiary_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/author>'):
                author_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/starring>'):
                starring_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/spouse>'):
                spouse_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/foundationPlace>'):
                foundationPlace_prop.append(line)
            elif str(line[1]).__eq__('<http://dbpedia.org/ontology/birthPlace>'):
                birthPlace_prop.append(line)
            else:
                count = count - 1
                exit(1)

        if typ == "train":
            typ = ""
        elif typ == "test":
            typ = "_test"
        else:
            exit(1)
        self.getSentenceEmbeddings(self, data_dir,
                                   deathplace_prop,
                                   model,
                                   true_statements_embeddings,
                                   "deathPlace"+typ, n, True)
        self.getSentenceEmbeddings(self, data_dir,
                                   award_prop,
                                   model,
                                   true_statements_embeddings,
                                   "award"+typ, n, True)
        self.getSentenceEmbeddings(self, data_dir,
                                   subsidiary_prop,
                                   model,
                                   true_statements_embeddings,
                                   "subsidiary"+typ, n, True)
        self.getSentenceEmbeddings(self, data_dir,
                                   author_prop,
                                   model,
                                   true_statements_embeddings,
                                   "author"+typ, n, True)
        self.getSentenceEmbeddings(self, data_dir,
                                   starring_prop,
                                   model,
                                   true_statements_embeddings,
                                   "starring"+typ, n, True)
        self.getSentenceEmbeddings(self, data_dir,
                                   spouse_prop,
                                   model,
                                   true_statements_embeddings,
                                   "spouse"+typ, n, True)
        self.getSentenceEmbeddings(self, data_dir,
                                   foundationPlace_prop,
                                   model,
                                   true_statements_embeddings,
                                   "foundationPlace"+typ, n, True)
        self.getSentenceEmbeddings(self, data_dir,
                                   birthPlace_prop,
                                   model,
                                   true_statements_embeddings,
                                   "birthPlace"+typ, n, True)
    @staticmethod
    def getSentenceEmbeddings(self,data_dir, data, model, true_statements_embeddings, type = 'default', n =3, props = False, bpdp = False, web_ids = False, copaal=False):
        embeddings_textual_evedences = []
        website_ids = set()
        for idx, (s, p, o, c, p2, t1) in enumerate(data):
            # p = self.entityToNLrepresentation(self, p)
            # path = data_dir + "data/test/" + type.replace("_test", "") + "/"
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
            # ##########################################################################comment block to save website ids#########
            if web_ids == False:
                sentence_embeddings = dict()


                if (s+' '+p+' '+o) not in true_statements_embeddings.keys():
                    if bpdp == True:
                        temp22, temp33 = select_top_n_sentences(temp2, n, temp, type.replace("_test",""))
                    elif props == False:
                        temp22, temp33 = select_top_n_sentences(temp2, n, temp,type)
                    else:
                        temp22, temp33 = select_top_n_sentences(temp2, n, temp, type.replace("_test",""),type)

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
        #     comment above block and uncomment below block to save all website ids###########################################
        if web_ids == True:
            with open('../dataset/pg_ranks/' + 'all_websites_ids_'+type.replace("_test","")+'.txt',"a") as f:
                for item in list(website_ids):
                    f.write("%s\n" % item)
            return  embeddings_textual_evedences, true_statements_embeddings
        else:
                    #################################################################################finish block here
            if bpdp == True:
                path = ""
                if type.__contains__("test"):
                    path = data_dir + "data/bpdp/" + "combined/"
                    self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences, "test", n)
                else:
                    path = data_dir + "data/bpdp/" + "combined/"
                    self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences, "train", n)
            elif props == False:
                path = ""
                if copaal:
                    data_dir1 = data_dir+"data/copaal/"
                else:
                    data_dir1 = data_dir + "data/"
                if type.__contains__("test"):
                    path = data_dir1 + "test/" + type.replace("_test","") + "/"
                    self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences, "test", n)
                else:
                    path = data_dir1 + "train/" + type + "/"
                    self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences, "train", n)
            elif type.__contains__("complete"):
                path = ""
                if type.__contains__("test"):
                    path = data_dir + "complete_dataset/"
                    self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences, "test", n)
                else:
                    path = data_dir + "complete_dataset/"
                    self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences, "train", n)
            else:
                path = ""
                if type.__contains__("test"):
                    path = data_dir + "properties_split/test/" + type.replace("_test","") + "/"
                    self.saveSentenceEmbeddingToFile(self, path, embeddings_textual_evedences, "test", n)
                else:
                    path = data_dir + "properties_split/train/" + type + "/"
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

        return "<"+quote(s)+">", "<"+quote(p)+">", "<"+quote(o)+">"

    @staticmethod
    def extract_sentence_embeddings_from_factcheck_output_multiclass(self,data_dir=None, copaal = False):
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
                    s, p, o = unquote(s), unquote(p), unquote(o)
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
                    # skip awards relation
                    # if p.__contains__("/ontology/award"):
                    #     continue
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
        embeddings_textual_evedences_train_date, true_statements_embeddings = self.getSentenceEmbeddings(self, data_dir, date_data_train, model, true_statements_embeddings, 'date', n, full_hybrid=copaal)

        embeddings_textual_evedences_train_domain, true_statements_embeddings = self.getSentenceEmbeddings(self, data_dir, domain_data_train, model, true_statements_embeddings, 'domain', n, full_hybrid=copaal)

        embeddings_textual_evedences_train_domainrange, true_statements_embeddings = self.getSentenceEmbeddings(self, data_dir, domainrange_data_train, model, true_statements_embeddings, 'domainrange', n, full_hybrid=copaal)

        embeddings_textual_evedences_train_mix, true_statements_embeddings = self.getSentenceEmbeddings(self, data_dir, mix_data_train, model, true_statements_embeddings, 'mix', n, full_hybrid=copaal)

        embeddings_textual_evedences_train_property, true_statements_embeddings = self.getSentenceEmbeddings(self, data_dir, property_data_train, model, true_statements_embeddings, 'property', n, full_hybrid=copaal)

        embeddings_textual_evedences_train_random,true_statements_embeddings = self.getSentenceEmbeddings(self, data_dir, random_data_train, model, true_statements_embeddings, 'random', n, full_hybrid=copaal)

        embeddings_textual_evedences_train_range,true_statements_embeddings = self.getSentenceEmbeddings(self, data_dir, range_data_train, model, true_statements_embeddings, 'range', n, full_hybrid=copaal)

        true_statements_embeddings_test = {}
        embeddings_textual_evedences_test_date, true_statements_embeddings_test = self.getSentenceEmbeddings(self, data_dir, date_data_test, model, true_statements_embeddings_test, 'date_test', n, full_hybrid=copaal)
        embeddings_textual_evedences_test_domain,true_statements_embeddings_test = self.getSentenceEmbeddings(self, data_dir, domain_data_test, model, true_statements_embeddings_test, 'domain_test', n, full_hybrid=copaal)
        embeddings_textual_evedences_test_domainrange, true_statements_embeddings_test = self.getSentenceEmbeddings(self, data_dir, domainrange_data_test, model, true_statements_embeddings_test, 'domainrange_test', n, full_hybrid=copaal)
        embeddings_textual_evedences_test_mix,true_statements_embeddings_test = self.getSentenceEmbeddings(self, data_dir, mix_data_test, model, true_statements_embeddings_test, 'mix_test', n, full_hybrid=copaal)
        embeddings_textual_evedences_test_property,true_statements_embeddings_test = self.getSentenceEmbeddings(self, data_dir, property_data_test, model, true_statements_embeddings_test, 'property_test', n, full_hybrid=copaal)
        embeddings_textual_evedences_test_random,true_statements_embeddings_test = self.getSentenceEmbeddings(self, data_dir, random_data_test, model, true_statements_embeddings_test, 'random_test', n, full_hybrid=copaal)
        embeddings_textual_evedences_test_range,true_statements_embeddings_test = self.getSentenceEmbeddings(self, data_dir, range_data_test, model, true_statements_embeddings_test, 'range_test', n, full_hybrid=copaal)

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
                    s, p, o = unquote(s), unquote(p), unquote(o)
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

                    # skip awards relation
                    # if p.__contains__("/ontology/award"):
                    #     continue
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
        collective_set = date_data_train + domain_data_train +domainrange_data_train + mix_data_train +property_data_train +random_data_train +range_data_train

        self.get_properties_split(self, collective_set, data_dir, model, "train", True)
        collective_set1 = date_data_test + domain_data_test + domainrange_data_test + mix_data_test + property_data_test + random_data_test + range_data_test
        self.get_properties_split(self, collective_set1, data_dir, model,"test", False)
        exit(1)


path_dataset_folder = '../dataset/'
mc = True
property_split = False
full_hybrid = True
# se = SentenceEmbeddings(data_dir=path_dataset_folder,multiclass=mc, multiprops=property_split)






dataset1 = "factbench" #factbench
dataset2 = "bpdp" #bpdp
datasets = [dataset1, dataset2]
path_dataset_folder = '../dataset/'

for d in datasets:
    if d=="factbench":
        dataset_file = "factbench_factcheckoutput.txt"
        # if full_hybrid:
        #     path_dataset_folder = '../dataset/data/copaal/'
        # else:
        path_dataset_folder = '../dataset/'
        # continue
        se = SentenceEmbeddings(data_dir=path_dataset_folder,multiclass=mc, multiprops=property_split, dataset_file=dataset_file, copaal=full_hybrid)
    elif d =="bpdp":
        dataset_file = "BPDP_factcheckoutput.txt"
        path_dataset_folder = '../dataset/'
        continue
        se = SentenceEmbeddings(data_dir=path_dataset_folder, multiclass=mc, multiprops=property_split, dataset_file=dataset_file, bpdp =True)
