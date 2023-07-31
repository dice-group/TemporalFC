from data import Data

path_dataset_folder = '../dataset/'
dataset = Data(data_dir=path_dataset_folder)

true_facts = []

for i in dataset.train_set:
    s, p, o, label = i
    if label == 0:
        true_facts.append(s + '\t' + p + '\t' + o + '\n')

with open('dataset/true_facts_train.txt', 'w') as writer:
    for t in true_facts:
        writer.write(t)
