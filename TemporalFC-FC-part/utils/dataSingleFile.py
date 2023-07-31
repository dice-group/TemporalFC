class DataSingleFile:
    def __init__(self, data_dir=None, subpath=None):

        # Quick workaround as we happen to have duplicate triples.
        # None if load complete data, otherwise load parts of dataset with folders in wrong directory.
        # if subpath==None:
        self.data = list((self.load_data(data_dir , data_type=subpath)))
        #     self.test_data = list((self.load_data(data_dir , data_type="test")))
        # else:
        #     self.train_set = list((self.load_data(data_dir+"data/train/"+subpath, data_type="train")))
        #     self.test_data = list((self.load_data(data_dir+"data/test/"+subpath, data_type="test")))

        # factcheck predictions on train and test data
        # if subpath==None:
        #     self.train_set_pred = list((self.load_data(data_dir , data_type="train_pred",pred=True)))
        #     self.test_data_pred = list((self.load_data(data_dir , data_type="test_pred",pred=True)))
        # else:
        #     self.train_set_pred = list((self.load_data(data_dir+"data/train/"+subpath, data_type="train_pred",pred=True)))
        #     self.test_data_pred = list((self.load_data(data_dir+"data/test/"+subpath, data_type="test_pred",pred=True)))

        # self.data = self.train_set + self.test_data
        self.entities = self.get_entities(self.data)

        self.relations = list(set(self.get_relations(self.data)))

        self.idx_entities = dict()
        self.idx_relations = dict()

        # Generate integer mapping
        for i in self.entities:
            self.idx_entities[i] = len(self.idx_entities)
        for i in self.relations:
            self.idx_relations[i] = len(self.idx_relations)

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
    def load_data(data_dir, data_type, pred=False):
        try:
            data = []
            if pred == False:
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
            else:
                with open("%s%s.txt" % (data_dir, data_type), "r") as f:
                    for datapoint in f:
                        datapoint = datapoint.split()
                        if len(datapoint) == 4:
                            s, p, o, label = datapoint
                            data.append((s, p, o, label))
                        elif len(datapoint) == 3:
                            s, p, label = datapoint
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
