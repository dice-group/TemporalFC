import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import zipfile



# this is a test file to generate sentecnce embeddings


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# extracting evedience sentence and generating embeddings from those evedence sentences and storing them in CSV file format
class SentenceEmbeddings:
    def __init__(self, data_dir=None,multiclass=True):

        self.data_file = "textResults4.txt"
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

        model = SentenceTransformer('nq-distilbert-base-v1')
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
                    o = so[1].split(" predicate ")[0].replace(" ", "_")
                    if o == "" : o = "DUMMY"
                    assert o != ""
                    p = so[1].split(" predicate ")[1].replace("\n", "")
                    # print("line:" + line+ ":"+ score + ":"+ str(correct))
                    sentences = []
                    trustworthiness = []
                    if line.__contains__("[ComplexProofs{"):
                        print("line:" + line + ":" + score + ":" + str(correct))
                        for idx, proof in enumerate(line.split("ComplexProofs{")):
                            if idx == 0:
                                continue
                            # print (str(idx) + ":" +proof)
                            p1 = proof.split("}")[0]
                            p2 = p1.split(", proofPhrase='")[1]
                            p3 = p2.split("', trustworthinessScore='")
                            print(str(idx) + ":" + p3[0] + "tr: " +p3[1][:-1])
                            sentences.append(p3[0])
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


        model = SentenceTransformer('nq-distilbert-base-v1')
        embeddings_textual_evedences_train = []
        embeddings_textual_evedences_test = []

        embeddings_textual_evedences_train_date = []
        embeddings_textual_evedences_train_domain = []
        embeddings_textual_evedences_train_domainrange = []
        embeddings_textual_evedences_train_mix = []
        embeddings_textual_evedences_train_property = []
        embeddings_textual_evedences_train_random = []
        embeddings_textual_evedences_train_range = []

        triple_emb = model.encode(s + " " + p + " " + o)
        # ///////////////////////////////////////////////////////////////////////////////train
        for idx, (s, p, o, c, p2,t1) in enumerate(date_data_train):
            print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
            p3 = p2.copy()
            highst = 0.0
            p2 = ""
            for idx2, ss in enumerate(p3):
                st1 = model.encode(ss)
                cos_sim = util.pytorch_cos_sim(triple_emb,st1)
                print(cos_sim)
                # if float(cos_sim) >= highst:
                p2 = p2+ ss+" "
                    # highst = float(cos_sim)
                # if cosine(triple_emb, st1) < 0.1:
                #     p2.remove(ss)
                # else:
                #     print(cosine(triple_emb, st1))
            # Sentences are encoded by calling model.encode()
            sentence_embeddings = model.encode(p2)

            # for s2 in p2:
            #     print("\n" + s2)
            # avg_embedding = np.mean(sentence_embeddings, axis=0)
            #
            # if (np.isnan(np.sum(avg_embedding))):
            #     avg_embedding = np.zeros(768, dtype=int)

            embeddings_textual_evedences_train_date.append(sentence_embeddings)

        for idx, (s, p, o, c, p2,t1) in enumerate(domain_data_train):
            print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
            p3 = p2.copy()
            highst = 0.0
            p2 = ""
            for idx2, ss in enumerate(p3):
                st1 = model.encode(ss)
                cos_sim = util.pytorch_cos_sim(triple_emb, st1)
                print(cos_sim)
                # if float(cos_sim) >= highst:
                p2 = p2 + ss + " "
                # highst = float(cos_sim)
                # if cosine(triple_emb, st1) < 0.1:
                #     p2.remove(ss)
                # else:
                #     print(cosine(triple_emb, st1))
            # Sentences are encoded by calling model.encode()
            sentence_embeddings = model.encode(p2)

            # for s2 in p2:
            #     print("\n" + s2)
            # avg_embedding = np.mean(sentence_embeddings, axis=0)
            #
            # if (np.isnan(np.sum(avg_embedding))):
            #     avg_embedding = np.zeros(768, dtype=int)

            embeddings_textual_evedences_train_domain.append(sentence_embeddings)

        for idx, (s, p, o, c, p2,t1) in enumerate(domainrange_data_train):
            print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
            p3 = p2.copy()
            highst = 0.0
            p2 = ""
            for idx2, ss in enumerate(p3):
                st1 = model.encode(ss)
                cos_sim = util.pytorch_cos_sim(triple_emb, st1)
                print(cos_sim)
                # if float(cos_sim) >= highst:
                p2 = p2 + ss + " "
                # highst = float(cos_sim)
                # if cosine(triple_emb, st1) < 0.1:
                #     p2.remove(ss)
                # else:
                #     print(cosine(triple_emb, st1))
            # Sentences are encoded by calling model.encode()
            sentence_embeddings = model.encode(p2)

            # for s2 in p2:
            #     print("\n" + s2)
            # avg_embedding = np.mean(sentence_embeddings, axis=0)
            #
            # if (np.isnan(np.sum(avg_embedding))):
            #     avg_embedding = np.zeros(768, dtype=int)

            embeddings_textual_evedences_train_domainrange.append(sentence_embeddings)

        for idx, (s, p, o, c, p2,t1) in enumerate(mix_data_train):
            print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")

            p3 = p2.copy()
            highst = 0.0
            p2 = ""
            for idx2, ss in enumerate(p3):
                st1 = model.encode(ss)
                cos_sim = util.pytorch_cos_sim(triple_emb, st1)
                print(cos_sim)
                # if float(cos_sim) >= highst:
                p2 = p2 + ss + " "
                # highst = float(cos_sim)
                # if cosine(triple_emb, st1) < 0.1:
                #     p2.remove(ss)
                # else:
                #     print(cosine(triple_emb, st1))
            # Sentences are encoded by calling model.encode()
            sentence_embeddings = model.encode(p2)

            # for s2 in p2:
            #     print("\n" + s2)
            # avg_embedding = np.mean(sentence_embeddings, axis=0)
            #
            # if (np.isnan(np.sum(avg_embedding))):
            #     avg_embedding = np.zeros(768, dtype=int)

            embeddings_textual_evedences_train_mix.append(sentence_embeddings)

        for idx, (s, p, o, c, p2,t1) in enumerate(property_data_train):
            print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
            p3 = p2.copy()
            highst = 0.0
            p2 = ""
            for idx2, ss in enumerate(p3):
                st1 = model.encode(ss)
                cos_sim = util.pytorch_cos_sim(triple_emb, st1)
                print(cos_sim)
                # if float(cos_sim) >= highst:
                p2 = p2 + ss + " "
                # highst = float(cos_sim)
                # if cosine(triple_emb, st1) < 0.1:
                #     p2.remove(ss)
                # else:
                #     print(cosine(triple_emb, st1))
            # Sentences are encoded by calling model.encode()
            sentence_embeddings = model.encode(p2)

            # for s2 in p2:
            #     print("\n" + s2)
            # avg_embedding = np.mean(sentence_embeddings, axis=0)
            #
            # if (np.isnan(np.sum(avg_embedding))):
            #     avg_embedding = np.zeros(768, dtype=int)

            embeddings_textual_evedences_train_property.append(sentence_embeddings)

        for idx, (s, p, o, c, p2,t1) in enumerate(random_data_train):
            print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
            p3 = p2.copy()
            highst = 0.0
            p2 = ""
            for idx2, ss in enumerate(p3):
                st1 = model.encode(ss)
                cos_sim = util.pytorch_cos_sim(triple_emb, st1)
                print(cos_sim)
                # if float(cos_sim) >= highst:
                p2 = p2 + ss + " "
                # highst = float(cos_sim)
                # if cosine(triple_emb, st1) < 0.1:
                #     p2.remove(ss)
                # else:
                #     print(cosine(triple_emb, st1))
            # Sentences are encoded by calling model.encode()
            sentence_embeddings = model.encode(p2)

            # for s2 in p2:
            #     print("\n" + s2)
            # avg_embedding = np.mean(sentence_embeddings, axis=0)
            #
            # if (np.isnan(np.sum(avg_embedding))):
            #     avg_embedding = np.zeros(768, dtype=int)

            embeddings_textual_evedences_train_random.append(sentence_embeddings)

        for idx, (s, p, o, c, p2,t1) in enumerate(range_data_train):
            print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
            p3 = p2.copy()
            highst = 0.0
            p2 = ""
            for idx2, ss in enumerate(p3):
                st1 = model.encode(ss)
                cos_sim = util.pytorch_cos_sim(triple_emb, st1)
                print(cos_sim)
                # if float(cos_sim) >= highst:
                p2 = p2 + ss + " "
                # highst = float(cos_sim)
                # if cosine(triple_emb, st1) < 0.1:
                #     p2.remove(ss)
                # else:
                #     print(cosine(triple_emb, st1))
            # Sentences are encoded by calling model.encode()
            sentence_embeddings = model.encode(p2)

            # for s2 in p2:
            #     print("\n" + s2)
            # avg_embedding = np.mean(sentence_embeddings, axis=0)
            #
            # if (np.isnan(np.sum(avg_embedding))):
            #     avg_embedding = np.zeros(768, dtype=int)

            embeddings_textual_evedences_train_range.append(sentence_embeddings)

        # ///////////////////////////////////////////////////////////////////////////////test
        embeddings_textual_evedences_test_date = []
        embeddings_textual_evedences_test_domain = []
        embeddings_textual_evedences_test_domainrange = []
        embeddings_textual_evedences_test_mix = []
        embeddings_textual_evedences_test_property = []
        embeddings_textual_evedences_test_random = []
        embeddings_textual_evedences_test_range = []
        for idx, (s, p, o, c, p2,t1) in enumerate(date_data_test):
            print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")

            p3 = p2.copy()
            highst = 0.0
            p2 = ""
            for idx2, ss in enumerate(p3):
                st1 = model.encode(ss)
                cos_sim = util.pytorch_cos_sim(triple_emb, st1)
                print(cos_sim)
                # if float(cos_sim) >= highst:
                p2 = p2 + ss + " "
                # highst = float(cos_sim)
                # if cosine(triple_emb, st1) < 0.1:
                #     p2.remove(ss)
                # else:
                #     print(cosine(triple_emb, st1))
            # Sentences are encoded by calling model.encode()
            sentence_embeddings = model.encode(p2)

            # for s2 in p2:
            #     print("\n" + s2)
            # avg_embedding = np.mean(sentence_embeddings, axis=0)
            #
            # if (np.isnan(np.sum(avg_embedding))):
            #     avg_embedding = np.zeros(768, dtype=int)

            embeddings_textual_evedences_test_date.append(sentence_embeddings)


        for idx, (s, p, o, c, p2,t1) in enumerate(domain_data_test):
            print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
            p3 = p2.copy()
            highst = 0.0
            p2 = ""
            for idx2, ss in enumerate(p3):
                st1 = model.encode(ss)
                cos_sim = util.pytorch_cos_sim(triple_emb, st1)
                print(cos_sim)
                # if float(cos_sim) >= highst:
                p2 = p2 + ss + " "
                # highst = float(cos_sim)
                # if cosine(triple_emb, st1) < 0.1:
                #     p2.remove(ss)
                # else:
                #     print(cosine(triple_emb, st1))
            # Sentences are encoded by calling model.encode()
            sentence_embeddings = model.encode(p2)

            # for s2 in p2:
            #     print("\n" + s2)
            # avg_embedding = np.mean(sentence_embeddings, axis=0)
            #
            # if (np.isnan(np.sum(avg_embedding))):
            #     avg_embedding = np.zeros(768, dtype=int)

            embeddings_textual_evedences_test_domain.append(sentence_embeddings)


        for idx, (s, p, o, c, p2,t1) in enumerate(domainrange_data_test):
            print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
            p3 = p2.copy()
            highst = 0.0
            p2 = ""
            for idx2, ss in enumerate(p3):
                st1 = model.encode(ss)
                cos_sim = util.pytorch_cos_sim(triple_emb, st1)
                print(cos_sim)
                # if float(cos_sim) >= highst:
                p2 = p2 + ss + " "
                # highst = float(cos_sim)
                # if cosine(triple_emb, st1) < 0.1:
                #     p2.remove(ss)
                # else:
                #     print(cosine(triple_emb, st1))
            # Sentences are encoded by calling model.encode()
            sentence_embeddings = model.encode(p2)

            # for s2 in p2:
            #     print("\n" + s2)
            # avg_embedding = np.mean(sentence_embeddings, axis=0)
            #
            # if (np.isnan(np.sum(avg_embedding))):
            #     avg_embedding = np.zeros(768, dtype=int)

            embeddings_textual_evedences_test_domainrange.append(sentence_embeddings)

        for idx, (s, p, o, c, p2,t1) in enumerate(mix_data_test):
            print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
            # Sentences are encoded by calling model.encode()
            p3 = p2.copy()
            highst = 0.0
            p2 = ""
            for idx2, ss in enumerate(p3):
                st1 = model.encode(ss)
                cos_sim = util.pytorch_cos_sim(triple_emb, st1)
                print(cos_sim)
                # if float(cos_sim) >= highst:
                p2 = p2 + ss + " "
                # highst = float(cos_sim)
                # if cosine(triple_emb, st1) < 0.1:
                #     p2.remove(ss)
                # else:
                #     print(cosine(triple_emb, st1))
            # Sentences are encoded by calling model.encode()
            sentence_embeddings = model.encode(p2)

            # for s2 in p2:
            #     print("\n" + s2)
            # avg_embedding = np.mean(sentence_embeddings, axis=0)
            #
            # if (np.isnan(np.sum(avg_embedding))):
            #     avg_embedding = np.zeros(768, dtype=int)

            embeddings_textual_evedences_test_mix.append(sentence_embeddings)

        for idx, (s, p, o, c, p2,t1) in enumerate(property_data_test):
            print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
            # Sentences are encoded by calling model.encode()
            p3 = p2.copy()
            highst = 0.0
            p2 = ""
            for idx2, ss in enumerate(p3):
                st1 = model.encode(ss)
                cos_sim = util.pytorch_cos_sim(triple_emb, st1)
                print(cos_sim)
                # if float(cos_sim) >= highst:
                p2 = p2 + ss + " "
                # highst = float(cos_sim)
                # if cosine(triple_emb, st1) < 0.1:
                #     p2.remove(ss)
                # else:
                #     print(cosine(triple_emb, st1))
            # Sentences are encoded by calling model.encode()
            sentence_embeddings = model.encode(p2)
            #
            # for s2 in p2:
            #     print("\n" + s2)
            # avg_embedding = np.mean(sentence_embeddings, axis=0)
            #
            # if (np.isnan(np.sum(avg_embedding))):
            #     avg_embedding = np.zeros(768, dtype=int)

            embeddings_textual_evedences_test_property.append(sentence_embeddings)

        for idx, (s, p, o, c, p2,t1) in enumerate(random_data_test):
            print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
            p3 = p2.copy()
            highst = 0.0
            p2 = ""
            for idx2, ss in enumerate(p3):
                st1 = model.encode(ss)
                cos_sim = util.pytorch_cos_sim(triple_emb, st1)
                print(cos_sim)
                # if float(cos_sim) >= highst:
                p2 = p2 + ss + " "
                # highst = float(cos_sim)
                # if cosine(triple_emb, st1) < 0.1:
                #     p2.remove(ss)
                # else:
                #     print(cosine(triple_emb, st1))
            # Sentences are encoded by calling model.encode()
            sentence_embeddings = model.encode(p2)
            #
            # for s2 in p2:
            #     print("\n" + s2)
            # avg_embedding = np.mean(sentence_embeddings, axis=0)
            #
            # if (np.isnan(np.sum(avg_embedding))):
            #     avg_embedding = np.zeros(768, dtype=int)

            embeddings_textual_evedences_test_random.append(sentence_embeddings)

        for idx, (s, p, o, c, p2,t1) in enumerate(range_data_test):
            print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
            p3 = p2.copy()
            highst = 0.0
            p2 = ""
            for idx2, ss in enumerate(p3):
                st1 = model.encode(ss)
                cos_sim = util.pytorch_cos_sim(triple_emb, st1)
                print(cos_sim)
                # if float(cos_sim) >= highst:
                p2 = p2 + ss + " "
                # highst = float(cos_sim)
                # if cosine(triple_emb, st1) < 0.1:
                #     p2.remove(ss)
                # else:
                #     print(cosine(triple_emb, st1))
            # Sentences are encoded by calling model.encode()
            sentence_embeddings = model.encode(p2)

            # for s2 in p2:
            #     print("\n" + s2)
            # avg_embedding = np.mean(sentence_embeddings, axis=0)
            #
            # if (np.isnan(np.sum(avg_embedding))):
            #     avg_embedding = np.zeros(768, dtype=int)

            embeddings_textual_evedences_test_range.append(sentence_embeddings)

        # /////////////////////////////////////////////////////////////////////////////// normal

        for idx, (s, p, o, c, p2,t1) in enumerate(data_train):
            print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
            p3 = p2.copy()
            highst = 0.0
            p2 = ""
            for idx2, ss in enumerate(p3):
                st1 = model.encode(ss)
                cos_sim = util.pytorch_cos_sim(triple_emb, st1)
                print(cos_sim)
                # if float(cos_sim) >= highst:
                p2 = p2 + ss + " "
                # highst = float(cos_sim)
                # if cosine(triple_emb, st1) < 0.1:
                #     p2.remove(ss)
                # else:
                #     print(cosine(triple_emb, st1))
            # Sentences are encoded by calling model.encode()
            sentence_embeddings = model.encode(p2)
            #
            # for s2 in p2:
            #     print("\n" + s2)
            # avg_embedding = np.mean(sentence_embeddings, axis=0)
            #
            # if (np.isnan(np.sum(avg_embedding))):
            #     avg_embedding = np.zeros(768, dtype=int)

            embeddings_textual_evedences_train.append(sentence_embeddings)




        for idx, (s, p, o, c, p2,t1) in enumerate(data_test):
            print(s + "\t" + p + "\t" + o + "\t" + str(c) + "\t")
            p3 = p2.copy()
            highst = 0.0
            p2 = ""
            for idx2, ss in enumerate(p3):
                st1 = model.encode(ss)
                cos_sim = util.pytorch_cos_sim(triple_emb, st1)
                print(cos_sim)
                # if float(cos_sim) >= highst:
                p2 = p2 + ss + " "
                # highst = float(cos_sim)
                # if cosine(triple_emb, st1) < 0.1:
                #     p2.remove(ss)
                # else:
                #     print(cosine(triple_emb, st1))
            # Sentences are encoded by calling model.encode()
            sentence_embeddings = model.encode(p2)

            # for s2 in p2:
            #     print("\n" + s2)
            # avg_embedding = np.mean(sentence_embeddings, axis=0)
            #
            # print(avg_embedding)
            # if (np.isnan(np.sum(avg_embedding))):
            #     avg_embedding = np.zeros(768, dtype=int)
            #     print(avg_embedding)
            #     # exit(1)

            embeddings_textual_evedences_test.append(sentence_embeddings)

        # /////////////////////////////////////////////////////////////////////////////// saving part
        path = data_dir+ "data/train/date/"
        X = np.array(embeddings_textual_evedences_train_date)
        print(X.shape)
        X = pd.DataFrame(X)
        compression_opts = dict(method='zip', archive_name='trainSE.csv')
        X.to_csv(path + 'trainSE.zip', index=False, compression=compression_opts)
        with zipfile.ZipFile(path + 'trainSE.zip', 'r') as zip_ref:
            zip_ref.extractall(path)

        path = data_dir + "data/train/domain/"
        X = np.array(embeddings_textual_evedences_train_domain)
        print(X.shape)
        X = pd.DataFrame(X)
        compression_opts = dict(method='zip', archive_name='trainSE.csv')
        X.to_csv(path + 'trainSE.zip', index=False, compression=compression_opts)
        with zipfile.ZipFile(path +'trainSE.zip', 'r') as zip_ref:
            zip_ref.extractall(path)

        path = data_dir + "data/train/domainrange/"
        X = np.array(embeddings_textual_evedences_train_domainrange)
        print(X.shape)
        X = pd.DataFrame(X)
        compression_opts = dict(method='zip', archive_name='trainSE.csv')
        X.to_csv(path + 'trainSE.zip', index=False, compression=compression_opts)
        with zipfile.ZipFile(path+'trainSE.zip', 'r') as zip_ref:
            zip_ref.extractall(path)

        path = data_dir + "data/train/mix/"
        X = np.array(embeddings_textual_evedences_train_mix)
        print(X.shape)
        X = pd.DataFrame(X)
        compression_opts = dict(method='zip', archive_name='trainSE.csv')
        X.to_csv(path + 'trainSE.zip', index=False, compression=compression_opts)
        with zipfile.ZipFile(path +'trainSE.zip', 'r') as zip_ref:
            zip_ref.extractall(path)

        path = data_dir + "data/train/property/"
        X = np.array(embeddings_textual_evedences_train_property)
        print(X.shape)
        X = pd.DataFrame(X)
        compression_opts = dict(method='zip', archive_name='trainSE.csv')
        X.to_csv(path + 'trainSE.zip', index=False, compression=compression_opts)
        with zipfile.ZipFile(path +'trainSE.zip', 'r') as zip_ref:
            zip_ref.extractall(path)

        path = data_dir + "data/train/random/"
        X = np.array(embeddings_textual_evedences_train_random)
        print(X.shape)
        X = pd.DataFrame(X)
        compression_opts = dict(method='zip', archive_name='trainSE.csv')
        X.to_csv(path + 'trainSE.zip', index=False, compression=compression_opts)
        with zipfile.ZipFile(path + 'trainSE.zip', 'r') as zip_ref:
            zip_ref.extractall(path)

        path = data_dir + "data/train/range/"
        X = np.array(embeddings_textual_evedences_train_range)
        print(X.shape)
        X = pd.DataFrame(X)
        compression_opts = dict(method='zip', archive_name='trainSE.csv')
        X.to_csv(path + 'trainSE.zip', index=False, compression=compression_opts)
        with zipfile.ZipFile(path +'trainSE.zip', 'r') as zip_ref:
            zip_ref.extractall(path)


        # /////

        path = data_dir + "data/test/date/"
        X = np.array(embeddings_textual_evedences_test_date)
        print(X.shape)
        X = pd.DataFrame(X)
        compression_opts = dict(method='zip', archive_name='testSE.csv')
        X.to_csv(path + 'testSE.zip', index=False, compression=compression_opts)
        with zipfile.ZipFile(path +'testSE.zip', 'r') as zip_ref:
            zip_ref.extractall(path)

        path = data_dir + "data/test/domain/"
        X = np.array(embeddings_textual_evedences_test_domain)
        print(X.shape)
        X = pd.DataFrame(X)
        compression_opts = dict(method='zip', archive_name='testSE.csv')
        X.to_csv(path + 'testSE.zip', index=False, compression=compression_opts)
        with zipfile.ZipFile(path +'testSE.zip', 'r') as zip_ref:
            zip_ref.extractall(path)

        path = data_dir + "data/test/domainrange/"
        X = np.array(embeddings_textual_evedences_test_domainrange)
        print(X.shape)
        X = pd.DataFrame(X)
        compression_opts = dict(method='zip', archive_name='testSE.csv')
        X.to_csv(path + 'testSE.zip', index=False, compression=compression_opts)
        with zipfile.ZipFile(path + 'testSE.zip', 'r') as zip_ref:
            zip_ref.extractall(path)

        path = data_dir + "data/test/mix/"
        X = np.array(embeddings_textual_evedences_test_mix)
        print(X.shape)
        X = pd.DataFrame(X)
        compression_opts = dict(method='zip', archive_name='testSE.csv')
        X.to_csv(path + 'testSE.zip', index=False, compression=compression_opts)
        with zipfile.ZipFile(path+'testSE.zip', 'r') as zip_ref:
            zip_ref.extractall(path)

        path = data_dir + "data/test/property/"
        X = np.array(embeddings_textual_evedences_test_property)
        print(X.shape)
        X = pd.DataFrame(X)
        compression_opts = dict(method='zip', archive_name='testSE.csv')
        X.to_csv(path + 'testSE.zip', index=False, compression=compression_opts)
        with zipfile.ZipFile(path+'testSE.zip', 'r') as zip_ref:
            zip_ref.extractall(path)

        path = data_dir + "data/test/random/"
        X = np.array(embeddings_textual_evedences_test_random)
        print(X.shape)
        X = pd.DataFrame(X)
        compression_opts = dict(method='zip', archive_name='testSE.csv')
        X.to_csv(path + 'testSE.zip', index=False, compression=compression_opts)
        with zipfile.ZipFile(path+'testSE.zip', 'r') as zip_ref:
            zip_ref.extractall(path)

        path = data_dir + "data/test/range/"
        X = np.array(embeddings_textual_evedences_test_range)
        print(X.shape)
        X = pd.DataFrame(X)
        compression_opts = dict(method='zip', archive_name='testSE.csv')
        X.to_csv(path + 'testSE.zip', index=False, compression=compression_opts)
        with zipfile.ZipFile(path+'testSE.zip', 'r') as zip_ref:
            zip_ref.extractall(path)
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






path_dataset_folder = '/home/umair/Documents/pythonProjects/HybridFactChecking/dataset/'
se = SentenceEmbeddings(data_dir=path_dataset_folder,multiclass=True)
