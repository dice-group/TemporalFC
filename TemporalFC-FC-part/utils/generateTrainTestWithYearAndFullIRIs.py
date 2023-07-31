class FileProcessor:
    def __init__(self, input_file, output_file1, output_file2):
        self.input_file = input_file
        self.output_file1 = output_file1
        self.output_file2 = output_file2
    def replace_predicate(self, pred):
        if pred.__contains__("commander"):
            pred.replace("commander","director")
        elif pred.__contains__("producer"):
            pred.replace("producer","commander")
        elif pred.__contains__("musicComposer"):
            pred.replace("musicComposer","artist")
        elif pred.__contains__("director"):
            pred.replace("director","architect")
        elif pred.__contains__("author"):
            pred.replace("author","musicComposer")
        elif pred.__contains__("artist"):
            pred.replace("artist","producer")
        elif pred.__contains__("architect"):
            pred.replace("architect","author")
        else:
            pred = "None"

        return pred

    def process_file(self):
        with open(self.input_file, 'r') as f:
            lines = f.readlines()
        with open('/home/umair/Documents/pythonProjects/TemporalFC-FC-part/dataset/complete_dataset/dbpedia34k/relations.dict', 'r') as f:
            relations = f.readlines()
        num_relations = (len(relations)+1)*2
        subjects = []
        objects = []
        predicates = []

        with open(self.output_file1, 'w') as f:
            for line in lines:
                elements = line.strip().split('\t')
                if len(elements) == 5:
                    subjects.append("<http://dbpedia.org/resource/"+elements[0]+">")
                    predicates.append("<http://dbpedia.org/ontology/"+elements[1]+">")
                    objects.append("<http://dbpedia.org/resource/"+elements[2]+">")
                    f.write("<http://dbpedia.org/resource/"+elements[0]+">\t"+"<http://dbpedia.org/ontology/"+elements[1]
                            +">\t"+ "<http://dbpedia.org/resource/"+elements[2]+">\t"
                            +"True"+"\t"+ elements[3]  +'\n')
        with open(self.output_file2, 'w') as f:
            for line in lines:
                elements = line.strip().split('\t')
                if len(elements) == 5:
                    pred = predicates.pop()
                    if pred == "None":
                        print("something is wrong")
                        break
                    pred = self.replace_predicate(pred)
                    f.write(objects.pop()+"\t"+pred+"\t"+ subjects.pop()+"\t"
                            +"False"+"\t"+ elements[3]  +'\n')


data_type = "train"

fp = FileProcessor('../dataset/dbpedia5/orignal_data/'+data_type+'_original.txt', '../dataset/dbpedia5/positive_'+data_type+'/'+data_type+'year.txt', '../dataset/dbpedia5/negative_'+data_type+'/'+data_type+'year.txt')
fp.process_file()
