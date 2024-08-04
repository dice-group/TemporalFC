from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pytorch_lightning as pl
import torch

class StandardDataModule(pl.LightningDataModule):
    """
    train, valid and test sets are available.
    """

    def __init__(self, train_set_idx, entities_count, relations_count, times_count, batch_size, form,
                 num_workers=32, valid_set_idx=None, test_set_idx=None, neg_sample_ratio=None):
        super().__init__()
        self.train_set_idx = train_set_idx
        self.valid_set_idx = valid_set_idx
        self.test_set_idx = test_set_idx

        self.num_entities = entities_count
        self.num_relations = relations_count
        self.num_times = times_count

        self.form = form
        # self.batch_size = batch_size
        self.num_workers = num_workers
        self.neg_sample_ratio = neg_sample_ratio
        if self.form == 'FactChecking':  # we can name it as FactChecking
            self.dataset_type_class = FactCheckingDataset
            self.target_dim = 1
            self.neg_sample_ratio = neg_sample_ratio
        elif self.form == 'TimePrediction':  # we can name it as FactChecking
            self.dataset_type_class = TimePredictionDataset
            self.target_dim = 1
            self.neg_sample_ratio = neg_sample_ratio
        else:
            raise ValueError

    # Train, Valid, TestDATALOADERs
    def train_dataloader(self, batch_size1) -> DataLoader:
        if self.form == 'FactChecking':
            self.batch_size = batch_size1
            train_set = FactCheckingDataset(self.train_set_idx,
                                            num_entities=(self.num_entities),
                                            num_relations=(self.num_relations),
                                            num_times=(self.num_times))
            return DataLoader(train_set, batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers)
        elif self.form == 'TimePrediction':
            self.batch_size = batch_size1
            train_set = TimePredictionDataset(self.train_set_idx,
                                            num_entities=(self.num_entities),
                                            num_relations=(self.num_relations),
                                            num_times=(self.num_times))
            return DataLoader(train_set, batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers)

    def val_dataloader(self, batch_size1) -> DataLoader:

        if self.form == 'FactChecking':
            self.batch_size = batch_size1
            val_set = FactCheckingDataset(self.valid_set_idx,
                                            num_entities=(self.num_entities),
                                            num_relations=(self.num_relations),
                                            num_times=(self.num_times))
            return DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        elif self.form == 'TimePrediction':
            self.batch_size = batch_size1
            val_set = TimePredictionDataset(self.valid_set_idx,
                                            num_entities=(self.num_entities),
                                            num_relations=(self.num_relations),
                                            num_times=(self.num_times))
            return DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


    def dataloaders(self, batch_size1) -> DataLoader:
        if self.form == 'FactChecking':
            test_set = FactCheckingDataset(self.test_set_idx,
                                               num_entities=(self.num_entities),
                                               num_relations=(self.num_relations),
                                               num_times=(self.num_times))
            return DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        elif self.form == 'TimePrediction':
            test_set = TimePredictionDataset(self.test_set_idx,
                                               num_entities=(self.num_entities),
                                               num_relations=(self.num_relations),
                                               num_times=(self.num_times))
            return DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def setup(self, *args, **kwargs):
        pass

    def transfer_batch_to_device(self, *args, **kwargs):
        pass

    def prepare_data(self, *args, **kwargs):
        # Nothing to be prepared for now.
        pass


class TimePredictionDataset(Dataset):
    """
    Similar Issue =
    https://github.com/pytorch/pytorch/issues/50089
    https://github.com/PyTorchLightning/pytorch-lightning/issues/538
    """
    def __init__(self, triples_idx, num_entities, num_relations, num_times, neg_sample_ratio=0):
        self.neg_sample_ratio = neg_sample_ratio  # 0 Implies that we do not add negative samples. This is needed during testing and validation
        triples_idx = torch.LongTensor(triples_idx)
        self.head_idx = triples_idx[:, 0]
        self.rel_idx = triples_idx[:, 1]
        self.tail_idx = triples_idx[:, 2]
        self.time_idx = triples_idx[:, 3]
        self.sent_idx = triples_idx[:, 4]
        self.score_idx = triples_idx[:, 5]
        self.lbl_idx = triples_idx[:, 6]

        # assert self.sent_idx == self.head_idx.shape == self.rel_idx.shape == self.tail_idx.shape == self.lbl_idx.shape == self.score_idx.shape == self.time_idx.shape
        assert self.head_idx.shape == self.rel_idx.shape == self.tail_idx.shape == self.lbl_idx.shape  == self.time_idx.shape
        self.length = len(triples_idx)

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_times = num_times

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        h = self.head_idx[idx]
        r = self.rel_idx[idx]
        t = self.tail_idx[idx]
        time = self.time_idx[idx]
        sent = self.sent_idx[idx]
        s = self.score_idx[idx]
        l = self.lbl_idx[idx]
        return h, r, t,time, sent, s, l

class FactCheckingDataset(Dataset):
    """
    Similar Issue =
    https://github.com/pytorch/pytorch/issues/50089
    https://github.com/PyTorchLightning/pytorch-lightning/issues/538
    """
    def __init__(self, triples_idx, num_entities, num_relations, num_times, neg_sample_ratio=0):
        self.neg_sample_ratio = neg_sample_ratio  # 0 Implies that we do not add negative samples. This is needed during testing and validation
        triples_idx = torch.LongTensor(triples_idx)
        self.head_idx = triples_idx[:, 0]
        self.rel_idx = triples_idx[:, 1]
        self.tail_idx = triples_idx[:, 2]
        self.time_idx = triples_idx[:, 3]
        self.sent_idx = triples_idx[:, 4]
        self.score_idx = triples_idx[:, 5]
        self.lbl_idx = triples_idx[:, 6]

        assert (self.sent_idx.shape == self.head_idx.shape == self.rel_idx.shape == self.tail_idx.shape ==
                self.lbl_idx.shape == self.score_idx.shape == self.time_idx.shape)
        self.length = len(triples_idx)

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_times = num_times

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        h = self.head_idx[idx]
        r = self.rel_idx[idx]
        t = self.tail_idx[idx]
        time = self.time_idx[idx]
        sent = self.sent_idx[idx]
        s = self.score_idx[idx]
        l = self.lbl_idx[idx]
        return h, r, t,time, sent, s, l

    class FactCheckingDataset(Dataset):
        """
        Similar Issue =
        https://github.com/pytorch/pytorch/issues/50089
        https://github.com/PyTorchLightning/pytorch-lightning/issues/538
        """

        def __init__(self, triples_idx, num_entities, num_relations, num_times, neg_sample_ratio=0):
            self.neg_sample_ratio = neg_sample_ratio  # 0 Implies that we do not add negative samples. This is needed during testing and validation
            triples_idx = torch.LongTensor(triples_idx)
            self.head_idx = triples_idx[:, 0]
            self.rel_idx = triples_idx[:, 1]
            self.tail_idx = triples_idx[:, 2]
            self.time_idx = triples_idx[:, 3]
            self.sent_idx = triples_idx[:, 4]
            self.score_idx = triples_idx[:, 5]
            self.lbl_idx = triples_idx[:, 6]

            assert self.head_idx.shape == self.rel_idx.shape == self.tail_idx.shape == self.lbl_idx.shape == self.score_idx.shape == self.time_idx.shape
            self.length = len(triples_idx)

            self.num_entities = num_entities
            self.num_relations = num_relations
            self.num_times = num_times

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            h = self.head_idx[idx]
            r = self.rel_idx[idx]
            t = self.tail_idx[idx]
            time = self.time_idx[idx]
            sent = self.sent_idx[idx]
            s = self.score_idx[idx]
            l = self.lbl_idx[idx]
            return h, r, t, time,sent, s, l



    # def collate_fn(self, batch):
    #     batch = torch.LongTensor(batch)
    #     h, r, t, time, veracity, label = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
    #     # h, r, t, label = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3]
    #     size_of_batch, _ = batch.shape
    #     assert size_of_batch > 0
    #     label = torch.ones((size_of_batch,))
    #     # # Generate Negative Triples
    #     corr = torch.randint(0, self.num_entities, (size_of_batch * self.neg_sample_ratio, 2))
    #     #
    #     # # 2.1 Head Corrupt:
    #     h_head_corr = corr[:, 0]
    #     r_head_corr = r.repeat(self.neg_sample_ratio, )
    #     t_head_corr = t.repeat(self.neg_sample_ratio, )
    #     label_head_corr = torch.zeros(len(t_head_corr), )
    #
    #     # 2.2. Tail Corrupt
    #     h_tail_corr = h.repeat(self.neg_sample_ratio, )
    #     r_tail_corr = r.repeat(self.neg_sample_ratio, )
    #     t_tail_corr = corr[:, 1]
    #     label_tail_corr = torch.zeros(len(t_tail_corr), )
    #     #
    #     # # 3. Stack True and Corrupted Triples
    #     # h = torch.cat((h, h_head_corr, h_tail_corr), 0)
    #     # r = torch.cat((r, r_head_corr, r_tail_corr), 0)
    #     # t = torch.cat((t, t_head_corr, t_tail_corr), 0)
    #     label = torch.cat((label, label_head_corr, label_tail_corr), 0)
    #
    #     return (h, r, t), label
