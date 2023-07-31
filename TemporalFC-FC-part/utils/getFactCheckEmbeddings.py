import pandas as pd
# from dask import dataframe as dd
from data import Data

import numpy as np
import zipfile

path_dataset_folder='../dataset/'
str2="date/"
dataset1 = Data(data_dir=path_dataset_folder,subpath=str2)
train_folder = "data/train/"+str2
test_folder = "data/test/"+str2
train_set = list((dataset1.load_data(path_dataset_folder+train_folder, data_type="train")))
test_set = list((dataset1.load_data(path_dataset_folder+test_folder, data_type="test")))


# df_entities = pd.read_csv('../Embeddings/ConEx_dbpedia/ConEx_entity_embeddings.csv', index_col=0,chunksize=100000)
# pd_df = pd.concat(df_entities)
chunk_size = 100000
df_result = []
for chunk in pd.read_csv('../Embeddings/ConEx/ConEx_entity_embeddings.csv', index_col=0, chunksize=chunk_size):
    print(chunk.head)
    try:
        for test in chunk.index:
            print(test.rsplit('/', 1)[-1])
            e1 = test.rsplit('/', 1)[-1]
            e1 = e1.replace('>', '')
            print(test)
            if dataset1.entities.__contains__(e1):
                print(e1)
                df_result.append(chunk.T[test])

            else:
                print("not found:"+e1)
    finally:
        print("error in: "+e1 + ":")
        print(chunk.head)


name = "finalEntityEmbeddings"
X = pd.DataFrame([list(l) for l in df_result]).stack().apply(pd.Series).reset_index(1, drop=True)
    # X=pd.DataFrame(train_combined_emb_set)
print(X.head)
compression_opts = dict(method='zip', archive_name=name + '.csv')
X.to_csv(path_dataset_folder + name + '.zip', index=False, compression=compression_opts)
with zipfile.ZipFile(path_dataset_folder + name + '.zip', 'r') as zip_ref:
    zip_ref.extractall(path_dataset_folder)

    # process(chunk)

print(df_result)
# start = time.time()
# df_entities = dd.read_csv('../Embeddings/ConEx_dbpedia/ConEx_entity_embeddings.csv')
# end = time.time()
# df_entities.groupby(df_entities['0']).mean().compute()
# print("Read csv with dask: ",(end-start),"sec")
# print(df_entities.compute(num_workers=8))
# print(df_entities.size())
exit(1)
df_relations = pd.read_csv('../Embeddings/ConEx/ConEx_relation_embeddings.csv', index_col=0)
