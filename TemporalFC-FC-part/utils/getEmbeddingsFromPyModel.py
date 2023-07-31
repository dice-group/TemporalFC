
import tensorflow as tf
def main():
    path_entities_dict="/home/umair/Documents/pythonProjects/HybridFactChecking2/HybridFactChecking1/HybridFactChecking/dataset/complete_dataset/dbpedia34k/entities.dict"
    path_relations_dict="/home/umair/Documents/pythonProjects/HybridFactChecking2/HybridFactChecking1/HybridFactChecking/dataset/complete_dataset/dbpedia34k/relations.dict"
    print("getting embeddings from pytorch model: Conex")
    entities = set()
    relations = set()
    with open(path_entities_dict, 'r') as f:
        for line in f:
            datapoint = line.split()
            if len(datapoint) == 2:
                entities.add("http://dbpedia.org/resource/"+datapoint[1])

    with open(path_relations_dict, 'r') as f:
        for line in f:
            datapoint = line.split()
            if len(datapoint) == 2:
                relations.add("http://dbpedia.org/ontology/"+datapoint[1])

    con_lis = tf.convert_to_tensor(entities)
    print("Convert list to tensor:", con_lis)
    con_lis = tf.convert_to_tensor(entities)
    print("Convert list to tensor:", relations)





if __name__ == "__main__":
    main()


