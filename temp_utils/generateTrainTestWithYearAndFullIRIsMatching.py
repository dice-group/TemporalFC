from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import string
class FileProcessor:
    def __init__(self, input_file, output_file1, output_file2, data_type):
        self.input_file = input_file
        self.output_file1 = output_file1
        self.output_file2 = output_file2
        self.data_type = data_type
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

    def remove_punctuation(self,  str1):
        punctuations = string.punctuation
        str7 = "".join(char for char in str1 if char not in punctuations)
        return str7
    def cosine_sim(self, str1, str2):
        # Initialize the count vectorizer
        count_vect = CountVectorizer()

        # Fit and transform the input strings
        str1_vect = count_vect.fit_transform([str1])
        str2_vect = count_vect.transform([str2])

        # Calculate the cosine similarity between the two vectors
        cosine = cosine_similarity(str1_vect, str2_vect)

        return cosine[0][0]

    def match_and_update(self):
        file1 = self.output_file1
        file2 = self.output_file2
        if str(file1).__contains__("test"):
            file3 = self.output_file2.replace("test","train")
        else:
            file3 = self.output_file2.replace("train", "test")
        # open the first file for reading
        with open(file1, 'r') as f1:
            # read all lines in the file and split each line into a list of values
            f1_lines = [line.strip().split('\t') for line in f1.readlines()]

            # open the first file for reading
        with open(file3, 'r') as f1:
                # read all lines in the file and split each line into a list of values
            f3_lines = [line.strip().split('\t') for line in f1.readlines()]

        # open the second file for reading
        with open(file2, 'r') as f2:
            # read all lines in the file and split each line into a list of values
            f2_lines = [line.strip().split('\t') for line in f2.readlines()]

        # create an empty list to store matching lines
        matching_lines = []
        thresold = 0.90
        i = 0
        # loop through each line in file 1
        for line2 in f1_lines:

            found = False
            # loop through each line in file 2

            for line1 in f2_lines:# check if the s, p, and o values in both lines match
                if self.remove_punctuation(line1[0]) == self.remove_punctuation(line2[0]) \
                        and self.remove_punctuation(line1[2]) ==  self.remove_punctuation(line2[2]):
                    # if the values match, add the s, p, o, t, and label to the matching_lines list
                    matching_lines.append([line2[0], line2[1], line2[2], line2[3], line1[3]])
                    # break out of the loop through file 2, since we found a match
                    found = True
                    break
                if self.remove_punctuation(line1[2]) == self.remove_punctuation(line2[0]) \
                        and self.remove_punctuation(line1[0]) ==  self.remove_punctuation(line2[2]):
                    # if the values match, add the s, p, o, t, and label to the matching_lines list
                    matching_lines.append([line2[0], line2[1], line2[2], line2[3], line1[3]])
                    # break out of the loop through file 2, since we found a match
                    found = True
                    break

            # if found==False:
            #     for line1 in f3_lines:
            #         # check if the s, p, and o values in both lines match
            #         if self.remove_punctuation(line1[0]) == self.remove_punctuation(line2[0]) \
            #                 and self.remove_punctuation(line1[2]) ==  self.remove_punctuation(line2[2]):
            #             # if the values match, add the s, p, o, t, and label to the matching_lines list
            #             matching_lines.append([line2[0], line2[1], line2[2], line2[3], line1[3]])
            #             # break out of the loop through file 2, since we found a match
            #             found = True
            #             break
            #         if self.remove_punctuation(line1[2]) == self.remove_punctuation(line2[0]) \
            #                 and self.remove_punctuation(line1[0]) ==  self.remove_punctuation(line2[2]):
            #             # if the values match, add the s, p, o, t, and label to the matching_lines list
            #             matching_lines.append([line2[0], line2[1], line2[2], line2[3], line1[3]])
            #             # break out of the loop through file 2, since we found a match
            #             found = True
            #             break
            i = i+1
            if found == False:
                print("problem with this triple:"+ str(line2))
                # exit(1)
            else:
                print("counter: "+str(i))
        # open a new file for writing the matching lines
        with open('../dataset/dbpedia5/orignal_data/'+self.data_type+'_matching_lines.txt', 'w') as f:
            # loop through each matching line and write it to the file
            for line in matching_lines:
                f.write('\t'.join(line) + '\n')
            print("done")

    def process_file(self):
        with open(self.input_file, 'r') as f:
            lines = f.readlines()
        with open('../data_TP/dbpedia34k/relations.dict', 'r') as f:
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
                    # f.write("<http://dbpedia.org/resource/"+elements[0]+">\t"+"<http://dbpedia.org/ontology/"+elements[1]
                    #         +">\t"+ "<http://dbpedia.org/resource/"+elements[2]+">\t"
                    #         +"True"+"\t"+ elements[3]  +'\n')
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


data_type = ["test","train"]
for typ in data_type:
    fp = FileProcessor('../data_TP/dbpedia5/orignal_data/'+typ+'1.txt', '../data_TP/dbpedia5/'+typ+'/'+typ+'1.txt', '../data_TP/dbpedia5/'+typ+'/'+typ+'_with_time_final2.txt', typ)
    fp.match_and_update()
# fp.process_file()
