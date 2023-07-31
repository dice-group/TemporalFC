import sys
import csv
import requests
import tarfile
import json
import gzip
def main():
    # arguments parameter
    api_upb = "http://unikge.cs.upb.de:5001"
    api_kgvec = "http://kgvec2go.org/rest/get-vector/dbpedia/"
    upb_api = True
    embedding_type = "TransE"
    dataset_names = ["bpdp","factbecnh","dbpedia34k"]
    selected_dataset = dataset_names[2]
    dbpb_dataset = False

    print("start filtering")
    if selected_dataset == "bpdp":
        path = "../Embeddings/"+embedding_type+"/bpdp/"
    elif selected_dataset == "factbench":
        path = "../Embeddings/"+embedding_type+"/"
    elif selected_dataset == "dbpedia34k":
        path = "../Embeddings/" + embedding_type + "/dbpedia34k/"
    else:
        print("please select a dataset first")

    path_data_train = ""
    path_data_test = ""
    if selected_dataset == "bpdp":
        path_data_train = "../dataset/data/bpdp/train/train.txt"
        path_data_test = "../dataset/data/bpdp/test/test.txt"
    elif selected_dataset == "factbench":
        path_data_train = "../dataset/complete_dataset/train.txt"
        path_data_test = "../dataset/complete_dataset/test.txt"
    elif selected_dataset == "dbpedia34k":
        path_data_train = "../dataset/complete_dataset/dbpedia34k/updated/train.txt"
        path_data_test = "../dataset/complete_dataset/dbpedia34k/updated/test.txt"
    else:
        print("please select an appropriate dataset first")
        # path_data = "../dataset/complete_dataset/train.txt"
    entities = set()
    relations = set()

    entities_emb = []
    relations_emb = []

    # reading all training entiies
    with open(path_data_train, 'r') as f:
        for line in f:
            datapoint = line.split()
            if len(datapoint) == 4:
                entities.add(datapoint[0])
                relations.add(datapoint[1])
                entities.add(datapoint[2])
            else:
                print(line)
                exit(1)

    # reading all testing entities
    with open(path_data_test, 'r') as f:
        for line in f:
            datapoint = line.split()
            if len(datapoint) == 4:
                entities.add(datapoint[0])
                relations.add(datapoint[1])
                entities.add(datapoint[2])
            else:
                print(line)
                exit(1)
    if upb_api:
        for ttt in relations:
            print(ttt)
            # ttt = ttt.replace("http://dbpedia.org", "")
            parameters = {
                "relations" : [ttt.replace(">","").replace("<","")],
                # "indexname": "complex_dbpedia_relation"
                "indexname": "transe_dbpedia_l2_relation"
                # "indexname": "transe_dbpedia_dot_relation"
            }
            headers = {
                'Content-type': 'application/json',
            }
            response = requests.get(api_upb + "/get-relation-embedding", data=json.dumps(parameters), headers=headers)

            if response.status_code==200:
                relations_emb.append(str(json.loads(response.content.decode('utf-8'))).replace('[',',').replace(']','')
                                     .replace('{\'','<').replace('\':','>').replace('}','').replace(" <embedding>","")
                                     .replace(", 'operator> 'lhs'","").replace(", 'operator> 'rhs'",""))
                print(response.status_code)
                print(json.loads(response.content.decode('utf-8')))
            else:
                print("can't find embeddings")
                print(response.status_code)
        for ttt in entities:
            print(ttt)
            # ttt = ttt.replace("http://dbpedia.org","")
            parameters = {
                "entities" : [ttt.replace(">","").replace("<","")],
                # "indexname": "complex_dbpedia_entity"
                "indexname": "transe_dbpedia_l2_entity"
                # "indexname": "transe_dbpedia_dot_entity"
            }
            headers = {
                'Content-type': 'application/json',
            }
            response = requests.get(api_upb+"/get-entity-embedding", data=json.dumps(parameters), headers=headers)

            if response.status_code==200:
                if str(json.loads(response.content.decode('utf-8'))).__contains__("resource/A_Connecticut_Yankee_in_King_Arthur"):
                    print("ok")
                emb_ent = str(json.loads(response.content.decode('utf-8'))).replace('[',',').replace(']','')\
                    .replace('{\'','<').replace('\':','>').replace('}','').replace('{"','<')\
                    .replace('":','>').replace(" <embedding> "," ")
                emb_ent = emb_ent.replace("<","<http://dbpedia.org")
                entities_emb.append(emb_ent)
                print(response.status_code)
                print(json.loads(response.content.decode('utf-8')))
            else:
                print("can't find embeddings")
                print(response.status_code)
                # exit(1)
        #
        #
        #
        #         # exit(1)
    else:
        for ttt in relations:
            print(ttt)
            # parameters = {
            #     "relations": [ttt.replace(">", "").replace("<", "")],
            #     "indexname": "complex_dbpedia_relation"
            #     # "indexname": "transe_dbpedia_dot_relation"
            # }
            headers = {
                'Content-type': 'application/json',
            }
            response = requests.get(api_kgvec +ttt.replace(">", "").replace("<", "").split("/")[-1], headers=headers)

            if response.status_code == 200:
                relations_emb.append("<"+str(json.loads(response.content.decode('utf-8'))['uri'])+">"+str(json.loads(response.content.decode('utf-8'))['vector']).replace("["," ,").replace("]",""))
                print(response.status_code)
                print(json.loads(response.content.decode('utf-8')))
            else:
                print("can't find embeddings")
                print(response.status_code)
                exit(1)

        for ttt in entities:
            print(ttt)
            # parameters = {
            #     "entities": [ttt.replace(">", "").replace("<", "")],
            #     # "indexname": "transe_dbpedia_complex_entity"
            #     # "indexname": "transe_dbpedia_dot_entity"
            # }
            headers = {
                'Content-type': 'application/json',
            }
            response = requests.get(api_kgvec +ttt.replace(">", "").replace("<", "").split("/")[-1], headers=headers)

            if response.status_code == 200:
                if str(json.loads(response.content.decode('utf-8'))).__contains__(
                        "resource/A_Connecticut_Yankee_in_King_Arthur"):
                    print("ok")
                if len(json.loads(response.content.decode('utf-8')))!=0:
                    entities_emb.append("<"+str(json.loads(response.content.decode('utf-8'))['uri'])+">"+
                                        str(json.loads(response.content.decode('utf-8'))['vector']).replace("["," ,").replace("]",""))
                    print(response.status_code)
                    print(json.loads(response.content.decode('utf-8')))
                else:
                    print("can't find embeddings")
                    print(response.status_code)
                    print(ttt)

                    # exit(1)

            else:
                print("can't find embeddings")
                print(response.status_code)
                print(ttt)
                # sleep(10)
                exit(1)


    ent_emb_dim = len(entities_emb[0].split(','))-1
    ent_first_row = ""
    ii =0
    while ii < ent_emb_dim:
        ent_first_row = ent_first_row  + ","+  str(ii)
        ii = ii +1
    rel_emb_dim = len(relations_emb[0].split(','))-1
    rel_first_row = ""
    ii = 0
    while ii < rel_emb_dim:
        rel_first_row = rel_first_row  + ","+  str(ii)
        ii = ii + 1
    with open(path+"all_entities_embeddings_final.txt", 'w') as writer:
        writer.write(ent_first_row+ "\n")
    with open(path+"all_relations_embeddings_final.txt", 'w') as writer:
        writer.write(rel_first_row+ "\n")


    with open(path+"all_entities_embeddings_final.txt", "a") as a_file:
        for key in entities_emb:
            a_file.write(key+ "\n")
    a_file.close()

    with open(path + "all_relations_embeddings_final.txt", "a") as a_file:
        for key in relations_emb:
            a_file.write(key+ "\n")



    # a_file.close()

    # with open(path+"", 'r') as f: ["/resource/Boeing_747_hull_losses"]
    #     for line in f:
    #         datapoint = line.split()
    #         if len(datapoint) == 4:
    #             entities.add(datapoint[0])
    #             relations.add(datapoint[1])
    #             entities.add(datapoint[2])
    #         else:
    #             print(line)
    #             exit(1)
    # with open(path +"", 'r') as f:
    #     for line in f:
    #         datapoint = line.split()
    #         if len(datapoint) == 4:
    #             entities.add(datapoint[0])
    #             relations.add(datapoint[1])
    #             entities.add(datapoint[2])
    #         else:
    #             print(line)
    #             exit(1)
    #
    #
    # print(len(entities))
    # result = []
    # count = 0
    # with gzip.open(file_emb, 'rb') as f:
    #     for line in f:
    #         print(str(line))
    #         if str(line).__contains__("<"):
    #             line =str(line).split("<")[1].replace("\\t","\t").replace("\\n","\n").replace("\\xc3\\xad","í")
    #             datapoint = line.split()
    #             if(datapoint[0].__contains__("resource")):
    #                 entity = datapoint[0].split("/resource/")
    #                 entity[-1] = entity[-1].replace("/", ".")
    #                 # print(entity)
    #                 # print(entity[-1][:-1])
    #                 if entities.__contains__(entity[-1][:-1]):
    #                     entity[-1] = entity[-1].replace(" ", "_").replace(",", "")
    #                     if datapoint[0].__contains__("dbpedia.org"):
    #                         if datapoint[0].__contains__("en.dbpedia.org") or datapoint[0].__contains__("//dbpedia.org") or datapoint[0].__contains__("//global.dbpedia.org"):
    #                             print(datapoint)
    #                             data = entity[-1][:-1]+" "+str(datapoint[1:]).replace("[",",").replace("]","").replace("'","").replace("\"","")
    #                             result.append(data)
    #                             count = count +1
    #                     else:
    #                         print(datapoint)
    #                         data = entity[-1][:-1] + " " + str(datapoint[1:]).replace("[", ",").replace("]", "").replace("'","").replace("\"","")
    #                         result.append(data)
    #         else:
    #             print("==================================>>>>>>>>>>>>>>>>>>>>>"+str(line))
    #         if len(result)>=100:
    #             with open(path+"entity_embedding_filter2.csv",'a') as f:
    #                 writer = csv.writer(f)
    #                 for l2 in result:
    #                     writer.writerow([l2])
    #             result.clear()
    #
    # print(count)
    # if len(result) >= 0:
    #     with open(path + "entity_embedding_filter2.csv", 'a') as f:
    #         writer = csv.writer(f)
    #         for l2 in result:
    #             writer.writerow([l2])
    #
    #     result.clear()
    # # relation embeddings
    #
    #
    # with open(file_emb_rel, 'r') as f:
    #     for line in f:
    #         datapoint = line.split()
    #         entity = datapoint[0].split("/")
    #         entity[-1] = entity[-1].replace(" ", "_").replace(",","")
    #         # print(entity)
    #         # print(entity[-1][:-1])
    #         if relations.__contains__(entity[-1][:-1]):
    #             if datapoint[0].__contains__("dbpedia.org"):
    #                 if datapoint[0].__contains__("en.dbpedia.org/ontology") or datapoint[0].__contains__("//dbpedia.org/ontology"):
    #                     if datapoint[1]=="rhs":
    #                         print(datapoint)
    #                         data = entity[-1][:-1]+" "+str(datapoint[5:]).replace("[",",").replace("]","").replace("'","")
    #                         result.append(data)
    #
    #             # else:
    #             #     print(datapoint)
    #             #     data = entity[-1][:-1] + " " + str(datapoint[5:]).replace("[", ",").replace("]", "").replace("'","")
    #             #     result.append(data)
    #
    #         if len(result)>=100:
    #             with open(path+"relation_embedding_filter2.csv",'a') as f:
    #                 writer = csv.writer(f)
    #                 for l2 in result:
    #                     writer.writerow([l2])
    #             result.clear()
    #
    #
    # if len(result) >= 0:
    #     with open(path + "relation_embedding_filter2.csv", 'a') as f:
    #         writer = csv.writer(f)
    #         for l2 in result:
    #             writer.writerow([l2])
    #
    #     result.clear()
    #
    #
    #
    #
    #

    exit(1)









if __name__ == "__main__":
    main()
