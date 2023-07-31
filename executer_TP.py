import warnings

import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score, classification_report, precision_score, auc
import nn_models_TP
from utils_TP.dataset_classes import StandardDataModule
from data_TP import Data
from utils_TP.static_funcs import *
import time
import torch
from sklearn.model_selection import KFold
# from pytorch_lightning_kfold.validation import KFoldCrossValidator

import json
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin,DataParallelPlugin
# from utils.dataset_classes import StandardDataModule
from pytorch_lightning import Trainer, seed_everything
seed_everything(42, workers=True)


class Execute_TP:
    def __init__(self, args):
        args = preprocesses_input_args(args)
        sanity_checking_with_arguments(args)
        self.args = args
        if args.eval_dataset=="BPDP":
            bpdp = True
            args.path_dataset_folder += "/data_TP/bpdp"


        # if args.model == "full-Hybrid":
        #     args.path_dataset_folder += '/data/copaal'
        #     hybrid_app = True


        # 1. Create an instance of KG.
        self.args.dataset = Data(args=args)

        # 2. Create a storage path  + Serialize dataset object.
        self.storage_path = create_experiment_folder(folder_name=args.storage_path)
        # self.eval_model = True if self.args.eval == 1 else False

        # 3. Save Create folder to serialize data. This two numerical value will be used in embedding initialization.
        self.args.num_entities, self.args.num_relations, self.args.num_times = self.args.dataset.num_entities, self.args.dataset.num_relations, self.args.dataset.num_times


        # 4. Create logger
        self.logger = create_logger(name=self.args.model, p=self.storage_path)

        # 5. KGE related parameters


    def store(self, trained_model) -> None:
        """
        Store trained_model model and save embeddings into csv file.
        :param trained_model:
        :return:
        """
        self.logger.info('Store full model.')
        # Save Torch model.
        torch.save(trained_model.state_dict(), self.storage_path + '/model.pt')
        self.args.dataset = ""
        with open(self.storage_path + '/configuration.json', 'w') as file_descriptor:
            temp = vars(self.args)
            temp.pop('gpus')
            temp.pop('tpu_cores')
            json.dump(temp, file_descriptor)

        self.logger.info('Stored data.')


    def start(self) -> None:
        """
        Train and/or Evaluate Model
        Store Mode
        """
        start_time = time.time()
        # 1. Train and Evaluate
        trained_model = self.train_and_eval()
        # 2. Store trained model
        self.store(trained_model)
        #
        total_runtime = time.time() - start_time

        if 60 * 60 > total_runtime:
            message = f'{total_runtime / 60:.3f} minutes'
        else:
            message = f'{total_runtime / (60 ** 2):.3f} hours'

        self.logger.info(f'Total Runtime:{message}')

    def train_and_eval(self) -> nn_models_TP.BaseKGE:
        """
        Training and evaluation procedure
        """
        self.logger.info('--- Parameters are parsed for training ---')


        # trainer = pl.Trainer.from_argparse_args(Namespace(**dict(train_config)), early_stop_callback=early_stop_callback)

        # 3. Init ModelCheckpoint callback, monitoring 'val_loss'
        # self.args.enable_checkpointing = True
        self.args.checkpoint_callback = True
        if self.args.sub_dataset_path==None:
            pth = self.args.eval_dataset
        else:
            pth = self.args.eval_dataset + "-" + self.args.sub_dataset_path.replace('/','')
        mdl = self.args.model

        # saves a checkpint model file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
        checkpoint = ModelCheckpoint(
            monitor="avg_val_loss_per_epoch",
            dirpath=self.storage_path,
            filename="sample-{"+(str(mdl).lower())+"}--{"+(str(pth).lower())+"}--"+self.args.emb_type+"-{epoch:02d}-{val_loss:.3f}",
            save_top_k=1,
            mode="min",
        )
        early_stopping_callback = EarlyStopping(monitor="avg_val_loss_per_epoch", patience=10)
        # 1. Create Pytorch-lightning Trainer object from input configuration
        # print(torch.cuda.device_count())
        if torch.cuda.is_available():
            self.trainer = pl.Trainer.from_argparse_args(self.args, plugins=DataParallelPlugin(),
                                                     callbacks = [early_stopping_callback, checkpoint], gpus=torch.cuda.device_count())
        else:
            self.trainer = pl.Trainer.from_argparse_args(self.args, plugins=DataParallelPlugin(),
                                                     callbacks = [early_stopping_callback, checkpoint])
        # 2. Check whether validation and test datasets are available.
        if self.args.dataset.is_valid_test_available():
            trained_model = self.training()
        self.logger.info('--- Training is completed  ---')

        # print(self.args.checkpoint_callback.best_model_path)

        return trained_model

    def training(self):
        """
        Train models with KvsAll or NegativeSampling
        :return:
        """
        # 1. Select model and labelling : triple prediction.
        model, form_of_labelling = select_model(self.args)
        # if not self.args.batch_size:
        self.args.batch_size = int(len(self.args.dataset.idx_train_set) / 3) + 1
        # if not self.args.batch_size:
        self.args.val_batch_size = int(len(self.args.dataset.idx_valid_set) / 2) + 1

        self.args.fast_dev_run=False
        self.args.accumulate_grad_batches = self.args.batch_size
        self.args.deterministic=True

        self.logger.info(f' Standard training starts: {model.name}-labeling:{form_of_labelling}')
        # 2. Create training data.
        dataset = StandardDataModule(train_set_idx=self.args.dataset.idx_train_set,
                                     valid_set_idx=self.args.dataset.idx_valid_set,
                                     test_set_idx=self.args.dataset.idx_test_set,
                                     entities_count=self.args.dataset.num_entities,
                                     relations_count=self.args.dataset.num_relations,
                                     times_count=self.args.dataset.num_times,
                                     form=form_of_labelling,
                                     batch_size=self.args.batch_size,
                                     num_workers=self.args.num_workers)

        # 3. Display the selected model's architecture.
        self.logger.info(model)

        train_data = dataset.train_dataloader(batch_size1=self.args.batch_size)
        val_data = dataset.val_dataloader(batch_size1=self.args.val_batch_size)

        # Create a KFoldCrossValidator instance
        # validator = KFoldCrossValidator(model, train_data, val_data, k=5)
        # self.trainer.add_callback(validator)
        # 5. Train model
        self.trainer.fit(model, train_data,val_data)
        # 6. Test model on validation and test sets if possible.
        self.trainer.test(ckpt_path='best',test_dataloaders=dataset.dataloaders(len(self.args.dataset.idx_test_set)))
        self.evaluate(model, dataset.train_set_idx, 'Evaluation of Train data: ' + form_of_labelling)
        self.evaluate(model, dataset.test_set_idx, 'Evaluation of Test data: '+ form_of_labelling)
        return model

    def mrr_score2(self, predictions, labels):
        # Convert predictions and labels to numpy arrays
        predictions = np.array(predictions)
        labels = np.array(labels)

        # Compute the reciprocal rank for each query
        reciprocal_ranks = []
        for query_index in range(len(predictions)):
            # Get the prediction and label for the current query
            prediction = predictions[query_index]
            label = labels[query_index]

            # Find the rank of the highest ranked relevant item
            rank = np.where(prediction == label)[0][0] + 1
            reciprocal_rank = 1.0 / rank
            reciprocal_ranks.append(reciprocal_rank)

        # Return the mean of all the reciprocal ranks
        return np.mean(reciprocal_ranks)
    def mrr_score(self, y_true, y_pred):
        """
        Calculate MRR (Mean Reciprocal Rank) for a list of predictions.

        Parameters:
        y_true (array): An array of true target values.
        y_pred (array): An array of predicted target values.

        Returns:
        float: The MRR score.
        """
        ranks = []
        for yt, yp in zip(y_true, y_pred):
            rank = np.where(yp == yt)[0][0] + 1
            ranks.append(1 / rank)

        return np.mean(ranks)
    def evaluate(self, model, triple_idx, info):
        print("evaluation")
        model.eval()
        self.logger.info(info)
        self.logger.info(f'Num of triples {len(triple_idx)}')

        X_test = np.array(triple_idx)[:, :5]
        y_test = np.array(triple_idx)[:, -3]

        # label = model.time_embeddings(y_test)
        label = y_test
        X_test_tensor = torch.Tensor(X_test).long()
        Y_test_tensor = torch.Tensor(y_test).long()
        idx_s, idx_p, idx_o, t_idx, v_data = X_test_tensor[:, 0], X_test_tensor[:, 1], X_test_tensor[:, 2], X_test_tensor[:, 3], X_test_tensor[:, 4]
        # 2. Prediction score
        if info.__contains__("Test"):
            prob = model.forward_triples(idx_s, idx_p, idx_o, t_idx, v_data,type="test")
        else:
            prob = model.forward_triples(idx_s, idx_p, idx_o, t_idx, v_data)
        # pred = (prob > 0.5).float()
        pred = prob.data.detach().numpy()
        max_pred = np.argmax(pred, axis=1)
        idx, sort_pred= torch.sort(prob,dim=1,descending=True)

        test_mrr = self.mrr_score(label, sort_pred)
        self.logger.info(test_mrr)
        self.logger.info( accuracy_score(max_pred, label))
        # self.logger.info(classification_report(max_pred, label))


# true negatives are ignored
# fa
