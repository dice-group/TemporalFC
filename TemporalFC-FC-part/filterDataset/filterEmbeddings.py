import sys
import csv

import tarfile
import gzip
def main():
    print("start filtering, parameters sould be like this:")
    print("path of the files, embeddings of the entities, train triples, test triples, relation embeddings ")
    for arg in sys.argv[1:]:
        print(arg)
    path = sys.argv[1:][0]
    file_emb = sys.argv[1:][1]
    file_emb_rel = sys.argv[1:][4]
    file_train = sys.argv[1:][2]
    file_test = sys.argv[1:][3]
    with open(path + "entity_embedding_filter2.csv", 'w') as writer:
        print("")
    with open(path + "relation_embedding_filter2.csv", 'w') as writer:
        print("")

    entities = set()
    relations = set()
    with open(path+file_train, 'r') as f:
        for line in f:
            datapoint = line.split()
            if len(datapoint) == 4:
                entities.add(datapoint[0])
                relations.add(datapoint[1])
                entities.add(datapoint[2])
            else:
                print(line)
                exit(1)
    with open(path + file_test, 'r') as f:
        for line in f:
            datapoint = line.split()
            if len(datapoint) == 4:
                entities.add(datapoint[0])
                relations.add(datapoint[1])
                entities.add(datapoint[2])
            else:
                print(line)
                exit(1)


    print(len(entities))
    result = []
    count = 0
    with gzip.open(file_emb, 'rb') as f:
        for line in f:
            if str(line).__contains__("<"):
                line =str(line).split("<")[1].replace("\\t","\t").replace("\\n","\n").replace("\\xc3\\xad","í")
                datapoint = line.split()
                if(datapoint[0].__contains__("resource")):
                    entity = datapoint[0].split("/resource/")
                    entity[-1] = entity[-1].replace("/", ".")
                    # print(entity)
                    # print(entity[-1][:-1])
                    if entities.__contains__(entity[-1][:-1]):
                        entity[-1] = entity[-1].replace(" ", "_").replace(",", "")
                        if datapoint[0].__contains__("dbpedia.org"):
                            if datapoint[0].__contains__("en.dbpedia.org") or datapoint[0].__contains__("//dbpedia.org") or datapoint[0].__contains__("//global.dbpedia.org"):
                                print(datapoint)
                                data = entity[-1][:-1]+" "+str(datapoint[1:]).replace("[",",").replace("]","").replace("'","").replace("\"","")
                                result.append(data)
                                count = count +1
                        else:
                            print(datapoint)
                            data = entity[-1][:-1] + " " + str(datapoint[1:]).replace("[", ",").replace("]", "").replace("'","").replace("\"","")
                            result.append(data)
            else:
                print(str(line))
            if len(result)>=100:
                with open(path+"entity_embedding_filter2.csv",'a') as f:
                    writer = csv.writer(f)
                    for l2 in result:
                        writer.writerow([l2])
                result.clear()

    print(count)
    if len(result) >= 0:
        with open(path + "entity_embedding_filter2.csv", 'a') as f:
            writer = csv.writer(f)
            for l2 in result:
                writer.writerow([l2])

        result.clear()
    # relation embeddings


    with open(file_emb_rel, 'r') as f:
        for line in f:
            datapoint = line.split()
            entity = datapoint[0].split("/")
            entity[-1] = entity[-1].replace(" ", "_").replace(",","")
            # print(entity)
            # print(entity[-1][:-1])
            if relations.__contains__(entity[-1][:-1]):
                if datapoint[0].__contains__("dbpedia.org"):
                    if datapoint[0].__contains__("en.dbpedia.org/ontology") or datapoint[0].__contains__("//dbpedia.org/ontology"):
                        if datapoint[1]=="rhs":
                            print(datapoint)
                            data = entity[-1][:-1]+" "+str(datapoint[5:]).replace("[",",").replace("]","").replace("'","")
                            result.append(data)

                # else:
                #     print(datapoint)
                #     data = entity[-1][:-1] + " " + str(datapoint[5:]).replace("[", ",").replace("]", "").replace("'","")
                #     result.append(data)

            if len(result)>=100:
                with open(path+"relation_embedding_filter2.csv",'a') as f:
                    writer = csv.writer(f)
                    for l2 in result:
                        writer.writerow([l2])
                result.clear()


    if len(result) >= 0:
        with open(path + "relation_embedding_filter2.csv", 'a') as f:
            writer = csv.writer(f)
            for l2 in result:
                writer.writerow([l2])

        result.clear()






    exit(1)




















if __name__ == "__main__":
    main()
