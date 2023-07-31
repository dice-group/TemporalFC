import warnings
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report
import nn_models
from data import Data
from utils.static_funcs import *
import time
import torch
import json
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin,DataParallelPlugin
from utils.dataset_classes import StandardDataModule
from pytorch_lightning import Trainer, seed_everything
seed_everything(42, workers=True)


class Execute:
    def __init__(self, args):
        args = preprocesses_input_args(args)
        sanity_checking_with_arguments(args)
        self.args = args
        if args.eval_dataset=="BPDP":
            bpdp = True
            # args.path_dataset_folder += "/data/bpdp"

        if args.eval_dataset=="Dbpedia5" and ((str(args.model).lower().__contains__("full-hybrid")) or (str(args.model).lower().__contains__("path"))) :
            args.model = "text-KGE-Hybrid"

        if (str(args.model).lower().__contains__("full-hybrid")) or (str(args.model).lower().__contains__("path")):# == "kge-path-hybrid":
            args.path_dataset_folder += 'hybrid_data/copaal/'

        # 1. Create an instance of KG.
        self.args.dataset = Data(args=args)

        # 2. Create a storage path  + Serialize dataset object.
        self.storage_path = create_experiment_folder(folder_name=args.storage_path)
        # self.eval_model = True if self.args.eval == 1 else False

        # 3. Save Create folder to serialize data. This two numerical value will be used in embedding initialization.
        if args.model!="text-only":
            self.args.num_entities, self.args.num_relations, self.args.num_times = self.args.dataset.num_entities, self.args.dataset.num_relations, self.args.dataset.num_times
        else:
            self.args.num_entities, self.args.num_relations = self.args.dataset.num_entities, self.args.dataset.num_relations


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

    def train_and_eval(self) -> nn_models.BaseKGE:
        """
        Training and evaluation procedure
        """
        self.logger.info('--- Parameters are parsed for training ---')


        # trainer = pl.Trainer.from_argparse_args(Namespace(**dict(train_config)), early_stop_callback=early_stop_callback)

        # 3. Init ModelCheckpoint callback, monitoring 'val_loss'
        # self.args.enable_checkpointing = True
        # self.args.checkpoint_callback = True
        if self.args.sub_dataset_path==None:
            pth = self.args.eval_dataset
        else:
            pth = self.args.eval_dataset + "-" + self.args.sub_dataset_path.replace('/','')
        mdl = self.args.model

        # saves a checkpint model file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
        checkpoint = ModelCheckpoint(
            monitor="avg_val_loss_per_epoch",
            dirpath=self.storage_path,
            filename="sample-{"+(str(mdl).lower())+"}--{"+(str(pth).lower())+"}-"+self.args.negative_triple_generation+"-"+self.args.emb_type+"-{epoch:02d}-{val_loss:.3f}",
            save_top_k=1,
            verbose=True,
            mode="min"
        )
        early_stopping_callback = EarlyStopping(monitor="avg_val_loss_per_epoch", patience=100)
        # 1. Create Pytorch-lightning Trainer object from input configuration
        # print(torch.cuda.device_count())
        if torch.cuda.is_available():
            self.trainer = pl.Trainer.from_argparse_args(self.args, strategy=DataParallelPlugin(),
                                                     callbacks = [checkpoint,early_stopping_callback], gpus=torch.cuda.device_count(),
                                                         log_every_n_steps = 3)
        else:
            self.trainer = pl.Trainer.from_argparse_args(self.args, strategy=DataParallelPlugin(),
                                                     callbacks = [checkpoint,early_stopping_callback],
                                                         log_every_n_steps = 3)
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
        self.args.batch_size = int(len(self.args.dataset.idx_train_data) / 3) + 1
        self.args.val_batch_size = int(len(self.args.dataset.idx_valid_data) / 2) + 1

        self.args.fast_dev_run=False
        self.args.accumulate_grad_batches = self.args.batch_size
        self.args.deterministic=True

        self.logger.info(f' Standard training starts: {model.name}-labeling:{form_of_labelling}')
        # 2. Create training data.
        # dataset = Data(args=self.args)
        dataset = StandardDataModule(train_set_idx=self.args.dataset.idx_train_data,
                                     valid_set_idx=self.args.dataset.idx_valid_data,
                                     test_set_idx=self.args.dataset.idx_test_data,
                                     entities_idx=self.args.dataset.idx_entities,
                                     relations_idx=self.args.dataset.idx_relations,
                                     times_idx=None,
                                     form=form_of_labelling,
                                     batch_size=self.args.batch_size,
                                     num_workers=self.args.num_workers)

        # 3. Display the selected model's architecture.
        self.logger.info(model)

        # 5. Train model
        self.trainer.fit(model, dataset.train_dataloader(batch_size1=self.args.batch_size),dataset.val_dataloader(batch_size1=self.args.val_batch_size))
        # 6. Test model on validation and test sets if possible.
        self.trainer.test(ckpt_path='best',dataloaders=dataset.dataloaders(len(self.args.dataset.idx_test_data)))
        self.evaluate(model, dataset.train_set_idx, 'Evaluation of Train data: ' + form_of_labelling)
        self.evaluate(model, dataset.test_set_idx, 'Evaluation of Test data: '+ form_of_labelling)
        return model

    def evaluate(self, model, triple_idx, info):
        print("evaluation")
        model.eval()
        self.logger.info(info)
        self.logger.info(f'Num of triples {len(triple_idx)}')

        X_test = np.array(triple_idx)[:, :5]
        y_test = np.array(triple_idx)[:, -2]
        print(y_test)
        X_test_tensor = torch.Tensor(X_test).long()
        idx_s, idx_p, idx_o,  x_data = X_test_tensor[:, 0], X_test_tensor[:, 1], X_test_tensor[:, 2], X_test_tensor[:, 4]
        # 2. Prediction score
        if info.__contains__("Test"):
            prob = model.forward_triples(idx_s, idx_p, idx_o,  x_data,type="test").flatten()
        else:
            prob = model.forward_triples(idx_s, idx_p, idx_o,  x_data).flatten()
        print(prob)
        pred = (prob > 0.50).float()

        pred = pred.data.detach().numpy()
        self.logger.info( accuracy_score(y_test, pred))
        self.logger.info(classification_report(y_test, pred))




