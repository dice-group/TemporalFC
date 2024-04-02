import json
import re
import pandas as pd
import zipfile
import pytorch_lightning as pl
import argparse
import os

# from sentence_transformers import SentenceTransformer
import numpy as np
import csv


current_dir = os.getcwd()
DATA_PATH = os.path.join(current_dir, "data_TP")

def argparse_default(description=None):
    parser = pl.Trainer.add_argparse_args(argparse.ArgumentParser())
    # Paths.
    parser.add_argument("--dataset_name", type=str, default='favel')
    parser.add_argument("--type", type=str, default='test')
    if description is None:
        return parser.parse_args()
    else:
        return parser.parse_args(description)
class VeracityExtractor:
    def __init__(self):
        print("Initializing VeracityExtractor")
        # self.model = SentenceTransformer(model_name)
        # self.vector_length = 768  # Length of each sentence vector


    def generate_triples_with_veracity_scores(self, input_file):
        i = 0
        triple_with_veracity_final = []
        sentences = []
        with open(input_file, 'r') as file:
            # sent_lines = file.readlines()
            for line in file:
                print(line)
                if line.startswith("{\"veracityValue\":"):
                    if not line.endswith("]\"}\n"):
                        lines = line
                        # Read lines until encountering end of json "]}"
                        while True:
                            line = file.readline()
                            if line.strip() == '-----------------------------':
                                break
                            lines = lines + line
                        line = lines
                    print("start extraction")
                    sentences.append(line)

        total = len(sentences)
        veracity_score = "N/A"
        for line in sentences:

            try:
                json_object = json.loads(line)
            except Exception as e:
                print(e)
            sentence_scores = []
            # Remove leading/trailing whitespaces and newline characters
            if "fact" in json_object.keys():
                print("copaal output")
                veracity_score = json_object['veracityValue'] #.split('\t')[:-1]
                fact = json_object['fact']
                subject = fact.split(", http:")[0][1:]
                predicate = "http:"+fact.split(", http:")[1]
                object = "http:"+fact.split(", http:")[2][:-1]


            else:
                # print("fact check output")
                veracity_score = json_object['defactoScore'] #.split('\t')[:-1]
                subject = json_object['subject']
                predicate = json_object['predicate']
                object = json_object['object']
            if veracity_score!="N/A":
                triple_with_veracity_final.append([subject,predicate,object,veracity_score])  # Reshape to have 3 vectors of length 786
            else:
                print("issue")
            #
            if i%500==0:
                print(str(i) + "/" + str(total))
            #     break
            i = i +1
            veracity_score = "N/A"
        # Convert the list of sentence vectors to a numpy array
        # sentence_vectors_final = np.vstack(sentence_vectors_final)
        return triple_with_veracity_final

    def write_to_file(self, vectors, output_file):
        with open(output_file, "w") as f:
            for item in vectors:
                f.write("<" + (item[0]) + ">\t<" + (item[1]) + ">\t<" + (item[2]) + ">\t" + str(item[3]) + "\t" + ".\n")
        print("data saved")
        print(f"Vectors written to '{output_file}' successfully.")

if __name__ == '__main__':

    # Create an instance of veracity extractor
    args = argparse_default()
    if args.dataset_name != None:
        dataset = args.dataset_name
    typ = args.type

    dataset_path = '../data_TP/'+dataset+'/'
    sbert_vectorizer = VeracityExtractor()

    # Provide the path to your input file
    # input_file_path = dataset_path+ typ+'/'+typ+'_sentences.txt'result_train_VT_true_pl_3.txt
    input_file_path = dataset_path + 'copaal_output/result_' + typ + '_COPAAL.txt'


    # Get sentence vectors
    vectors = sbert_vectorizer.generate_triples_with_veracity_scores(input_file_path)

    output_file_path = dataset_path +typ+'/'+ typ + '_v_scores.txt'
    sbert_vectorizer.write_to_file(vectors, output_file_path)
    # 'vectors' will contain the sentence vectors for each sentence in the file
    print(vectors)
