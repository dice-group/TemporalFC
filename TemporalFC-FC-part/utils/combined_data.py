import pandas as pd
class CData:
    def __init__(self, data_dir=None, subpath=None):

        # Quick workaround as we happen to have duplicate triples.
        # if subpath is none then all data
        if subpath==None:
            self.train_set = pd.read_csv(data_dir + "trainCombinedEmbeddings.csv")
            self.test_data = pd.read_csv(data_dir + "testCombinedEmbeddings.csv")
        else:
            self.train_set = pd.read_csv(data_dir+"data/train/"+subpath+"trainCombinedEmbeddings.csv")
            self.test_data = pd.read_csv(data_dir+"data/test/"+subpath+"testCombinedEmbeddings.csv")



        # self.data = self.train_set + self.test_data
        # self.entities = self.get_entities(self.data)

        # self.relations = list(set(self.get_relations(self.train_set) + self.get_relations(self.test_data)))

        # self.idx_entities = dict()
        # self.idx_relations = dict()
        #
        # # Generate integer mapping
        # for i in self.entities:
        #     self.idx_entities[i] = len(self.idx_entities)
        # for i in self.relations:
        #     self.idx_relations[i] = len(self.idx_relations)
        #
        # self.idx_train_data = []
        # for (s, p, o, label) in self.train_set:
        #     idx_s, idx_p, idx_o, label = self.idx_entities[s], self.idx_relations[p], self.idx_entities[o], label
        #     self.idx_train_data.append([idx_s, idx_p, idx_o, label])
        #
        # self.idx_test_data = []
        # for (s, p, o, label) in self.test_data:
        #     idx_s, idx_p, idx_o, label = self.idx_entities[s], self.idx_relations[p], self.idx_entities[o], label
        #     self.idx_test_data.append([idx_s, idx_p, idx_o, label])

    @staticmethod
    def load_data(data_dir, data_type):
        try:
            data = []
            with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                for datapoint in f:
                    datapoint = datapoint.split()
                    if len(datapoint) == 4:
                        s, p, o, label = datapoint
                        if label == 'True':
                            label = 1
                        else:
                            label = 0
                        data.append((s, p, o, label))
                    elif len(datapoint) == 3:
                        s, p, label = datapoint
                        assert label == 'True' or label == 'False'
                        if label == 'True':
                            label = 1
                        else:
                            label = 0
                        data.append((s, p, 'DUMMY', label))
                    else:
                        raise ValueError
        except FileNotFoundError as e:
            print(e)
            print('Add empty.')
            data = []
        return data

    @staticmethod
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities
