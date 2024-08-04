import numpy as np
import pandas as pd
import csv

import pytorch_lightning as pl
import argparse
import os



class SortandExcludeIncompleteTriples:
    def __init__(self):
        print("test")


    def sort_and_exclude_triples(self, path_triples, path_v_file,path_sentence_file):
        print("")
        # Read content from both files
        with open(path_triples, 'r') as file1:
            content1 = file1.readlines()

        with open(path_v_file, 'r') as file2:
            content2 = file2.readlines()

        with open(path_sentence_file, 'r') as file2:
            content3 = file2.readlines()

        veracity_dict = dict()
        # Extracting complete triples from the second file
        triples_set = set()
        for line in content2:
            columns = tuple(line.split('\t')[:3])
            triples_set.add(columns)
            veracity_dict[columns] = (line.split('\t')[3])

        embeddings_dict = dict()
        triples_set2 = set()
        for line in content3:
            columns2 = (line.split('","')[:3])
            if columns2[0][0] == '0':
                continue
            columns2[2] = columns2[2].split('",')[0]
            # columns2[2] = columns2[2][1:]
            columns2[0] = columns2[0][1:]
            if columns2[0] == '0':
                continue
            # for col in columns2:
            #     col = col.replace("\"","")
            #     columns.append(col)
            triples_set2.add(tuple(columns2))
            if len(line.split('",')) > 5:
                embeddings_dict[tuple(columns2)] = line.split('","')[3:]
                for i in range(len(embeddings_dict[tuple(columns2)])):
                    embeddings_dict[tuple(columns2)][i] = float(embeddings_dict[tuple(columns2)][i]
                                                                .replace("\n", "").replace("\\n", "").replace('"',
                                                                                                              '').replace(
                        '\'', '').replace(' ', ''))
            else:
                embeddings_dict[tuple(columns2)] = line.split('",')[3:]
                embeddings_dict[tuple(columns2)] = str(embeddings_dict[tuple(columns2)]).replace("\n", "").replace(
                    "\\n", "").replace('"', '').replace('\'', '').replace(' ', '')

        filtered_content1 = []
        filtered_triples = []
        filtered_content2 = []
        # Filter the rows in the first file based on complete triplets in file2
        for line in content1:
            if tuple(line.split('\t')[:3]) in triples_set and tuple(line.split('\t')[:3]) in triples_set2:
                print(line)
                filtered_content1.append(line)
                if tuple(line.split('\t')[:3]) not in embeddings_dict.keys():
                    continue
                # [tuple((line.split('\t')[:3],embeddings_dict[tuple(line.split('\t')[:3])]))]
                if not isinstance(embeddings_dict[tuple(line.split('\t')[:3])][0], float) and not isinstance(
                        embeddings_dict[tuple(line.split('\t')[:3])], list):
                    filtered_content2.append(
                        line.split('\t')[:3] + str(embeddings_dict[tuple(line.split('\t')[:3])])[2:-2].split(","))
                else:
                    filtered_content2.append(
                        line.split('\t')[:3] + str(embeddings_dict[tuple(line.split('\t')[:3])])[1:-1].split(","))
                v_score = veracity_dict[tuple(line.split('\t')[:3])]
                filtered_triples.append(tuple((line.split('\t')[:3], v_score)))

        self.write_to_csv(filtered_content2, path_sentence_file)

        with open(path_triples, 'w') as file1, open(path_v_file, 'w') as file2:
            for line in filtered_content1:
                file1.write(line)
            for line2 in filtered_triples:
                if not line2[1].__contains__("\n"):
                    file2.write("\t".join(line2[0]) + "\t" + line2[1] + "\n")
                else:
                    file2.write("\t".join(line2[0]) + "\t" + line2[1])


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

def argparse_default(description=None):
    parser = pl.Trainer.add_argparse_args(argparse.ArgumentParser())
    # Paths.
    parser.add_argument("--dataset_name", type=str, default='favel')
    parser.add_argument("--dataset_path", type=str, default='../data_TP/')
    parser.add_argument("--type", type=str, default='test')
    if description is None:
        return parser.parse_args()
    else:
        return parser.parse_args(description)

if __name__ == '__main__':
    print("Starting sorting and excluding incomplete triples!!")
    args = argparse_default()
    if args.dataset_name != None:
        dataset = args.dataset_name
    typ = args.type
    dataset_folder = args.dataset_path
    # dataset_folder = "../data_TP/"
    # dataset = "favel"
    # typ = "train"
    #
    path_v_file = dataset_folder + dataset + "/" + typ + "/" + typ + "_v_scores.txt"
    path_sentence_file = dataset_folder + dataset + "/" + typ + "/" + typ + "SE.csv"
    path_triples = dataset_folder + dataset + "/" + typ + "/" + typ
    sort_exclude_triples = SortandExcludeIncompleteTriples()
    sort_exclude_triples.sort_and_exclude_triples( path_triples=path_triples, path_v_file=path_v_file, path_sentence_file=path_sentence_file)
