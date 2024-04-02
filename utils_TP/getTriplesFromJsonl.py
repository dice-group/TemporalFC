import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import zipfile



# this class converts jsonl file to triples format for FEVER dataset

# extracting evedience sentence and generating embeddings from those evedence sentences and storing them in CSV file format
class JSONLtoNTConversion:
    def __init__(self, file_path=None, output_file_format='NT'):
        self.data_file = file_path
        self.format = output_file_format
        self.convert_jsonl_to_triples_format(self, self.data_file)


    @staticmethod
    def convert_jsonl_to_triples_format(self,data_dir=None):
        data = []

        with open(data_dir, "r") as file:
            for line in file:
                json_line = json.loads(line)
                print(json_line['triple'])
                data.append(json_line['triple'])


        with open(data_dir.replace(".jsonl", ".nt"), "w") as file:
            for triple in data:
                triple = triple.replace("/entity/","/prop/direct/").replace("/wiki/","/entity/")
                file.write('<'+triple.replace('\', \'','>\t<')[1:-1]+'>\t.\n')
        print('done')





if __name__ == '__main__':
    path_dataset_folder = '/home/umair/Desktop/NEBULA/FEVER/results/output_triples_IRIs_with_v_scores.jsonl'
    se = JSONLtoNTConversion(file_path=path_dataset_folder,output_file_format='NT')
