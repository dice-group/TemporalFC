import json
import re
import pandas as pd
import zipfile
import pytorch_lightning as pl
import argparse
import os
from pytorch_lightning import Trainer, seed_everything
seed_everything(42, workers=True)
# from huggingface_hub import REPO_ID_SEPARATOR

# Now you can use REPO_ID_SEPARATOR in your code
# print(REPO_ID_SEPARATOR)

from sentence_transformers import SentenceTransformer
import numpy as np
import csv

current_dir = os.getcwd()
DATA_PATH = os.path.join(current_dir, "data_TP")

def argparse_default(description=None):
    parser = pl.Trainer.add_argparse_args(argparse.ArgumentParser())
    # Paths.
    parser.add_argument("--dataset_name", type=str, default='factbench')
    parser.add_argument("--type", type=str, default='test')
    if description is None:
        return parser.parse_args()
    else:
        return parser.parse_args(description)

'''
This class is for
extracting evedience sentence and generating embeddings 
from factcheck output and storing them in CSV file format
'''
class SBERTVectorizer:
    def __init__(self, model_name='sentence-transformers/all-distilroberta-v1'):
        self.model = SentenceTransformer(model_name)
        self.vector_length = 768  # Length of each sentence vector


    def vectorize_sentences(self, input_file):
        i = 0
        final_concatenated_vectors = dict()
        with open(input_file, 'r') as file:
            sent_lines = file.readlines()
        sentences = []
        for line in sent_lines:
            if line.startswith('{'):
                sentences.append(line)
        sentence_vectors = []
        total = len(sentences)
        sentence_vectors_final = []
        final_triples = []
        for sentence in sentences:
            sentence2 = json.loads(sentence)
            sentence_scores = []
            # Remove leading/trailing whitespaces and newline characters
            sentence = sentence2['complexProofs'] #.split('\t')[:-1]
            for word in sentence:
                # print(word)
                # word = json.loads(word)
                sent = word['proofPhrase']
                score = word['pagerank']
                sentence_scores.append((sent, score))

            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            # Select top 3 sentences or pad with zeros if less than 3
            selected_sentences = sentence_scores[:3] if len(sentence_scores) >= 3 else (sentence_scores)
                                                                                        # + [('Zero padding', 0)] * (3 - len(sentence_scores)))

            sentence_vectors = []
            for sentence, _ in selected_sentences:
                vector = self.model.encode(sentence, convert_to_tensor=True)
                sentence_vectors.append(vector[:self.vector_length])  # Truncate or pad to required length

                # Fill remaining vectors with zeros if less than 3 sentences
            while len(sentence_vectors) < 3:
                sentence_vectors.append(np.zeros((self.vector_length,)))

            # Concatenate the sentence vectors into a single vector
            concatenated_vectors = np.hstack(sentence_vectors)

            # If the concatenated vector is shorter than required, pad with zeros
            if len(concatenated_vectors) < (self.vector_length * 3):
                padding_length = (self.vector_length * 3) - len(concatenated_vectors)
                concatenated_vectors = np.pad(concatenated_vectors, (0, padding_length), mode='constant')

            values_to_append = np.array(["<"+sentence2["subject"]+">", "<"+sentence2["predicate"]+">", "<"+sentence2["object"]+">"])
            object_array = np.empty((values_to_append.size + concatenated_vectors.shape[0],), dtype=object)
            object_array[:values_to_append.size] = values_to_append
            object_array[values_to_append.size:] = concatenated_vectors.astype(object)
            final_triples.append(object_array[:values_to_append.size])
            # final_concatenated_vectors = np.insert(concatenated_vectors, 0, values_to_append, axis=0)


            # final_concatenated_vectors["<"+sentence2["subject"]+">, <"+sentence2["predicate"]+">, <"+sentence2["object"]+">"] = concatenated_vectors
            sentence_vectors_final.append(object_array)  # Reshape to have 3 vectors of length 786
            #
            if i%500==0:
                print(str(i) + "/" + str(total))
            #     break
            i = i +1

            #
                # for sent, score in sentence_scores[:3]:
                #     if sent:
                #         vector = self.model.encode(word[''], convert_to_tensor=True)
                #         sentence_vectors.append(vector)

        # Convert the list of sentence vectors to a numpy array
        # sentence_vectors_final = np.vstack(sentence_vectors_final)
        return sentence_vectors_final, final_triples

    def write_to_csv(self, vectors, output_file):
        X = np.array(vectors)
        print(X.shape)
        firstline_numbers = []
        for j in range(X.shape[1]):
            firstline_numbers.append(str(j))
        X = pd.DataFrame(vectors)
        compression_opts = dict(method='zip', archive_name=output_file.split("/")[-1])
        X.to_csv(output_file.replace("csv","zip"), compression=compression_opts, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print("data saved")
        print(f"Vectors written to '{output_file}' successfully.")

    def write_to_file(self, vectors, output_file):
        with open(output_file, "w") as f:
            for item in vectors:
                f.write( (item[0]) + "\t" + (item[1]) + "\t" + (item[2]) + "\t" + ".\n")
        print("data saved")
        print(f"Vectors written to '{output_file}' successfully.")


if __name__ == '__main__':
    # Create an instance of SBERTVectorizer
    # data_TP / factbench / factcheck_output / result_FactCheck_train_FactCheck.txt result_FactCheck_train_FactCheck
    args = argparse_default()
    if args.dataset_name!=None:
        dataset = args.dataset_name
    dataset_path = '../data_TP/'+dataset+'/'
    typ = args.type
    sbert_vectorizer = SBERTVectorizer()

    # Provide the path to your input file
    # input_file_path = dataset_path+ typ+'/'+typ+'_sentences.txt'
    input_file_path = dataset_path  + 'factcheck_output/result_FactCheck_' + typ + '_FactCheck.txt'


    # Get sentence vectors
    vectors, final_triples = sbert_vectorizer.vectorize_sentences(input_file_path)

    output_file_path = dataset_path+typ+'/'+ typ+'SE.csv'
    sbert_vectorizer.write_to_csv(vectors, output_file_path)
    sbert_vectorizer.write_to_file(final_triples, output_file_path.replace('_sentences_vectors.csv','_triples.txt'))
    # 'vectors' will contain the sentence vectors for each sentence in the file
    print(vectors)
