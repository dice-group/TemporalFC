import os
import logging
import datetime
from scipy import stats
import matplotlib.pyplot as plt

from nn_models_TP.temporal_model import TemporalModel

import pytorch_lightning as pl
from typing import AnyStr, Tuple



def calculate_wilcoxen_score(df_data , typ):
    stats.probplot(df_data["emb-only"], dist="norm", plot=plt)
    plt.title(typ+" prediction Before Hybrid")
    plt.savefig(typ+"_prediction_Before_Hybrid.png")
    stats.probplot(df_data["hybrid"], dist="norm", plot=plt)
    plt.title(typ+" prediction after Hybrid")
    plt.savefig(typ+"_prediction_After_Hybrid.png")

    print(typ+"  difference:" + str(stats.wilcoxon(df_data["emb-only"], df_data["hybrid"], correction=True)))

def preprocesses_input_args(arg):
    # To update the default value of Trainer in pytorch-lightnings
    arg.max_epochs = arg.max_num_epochs
    del arg.max_num_epochs
    arg.check_val_every_n_epoch = arg.check_val_every_n_epochs
    del arg.check_val_every_n_epochs
    arg.num_processes = arg.num_workers
    arg.checkpoint_callback = False
    arg.find_unused_parameters = False
    arg.logger = False
    return arg

def create_experiment_folder(folder_name='Experiments'):
    directory = os.getcwd() + '/dataset/' + folder_name + '/'
    folder_name = str(datetime.datetime.now())
    path_of_folder = directory + folder_name
    os.makedirs(path_of_folder)
    return path_of_folder

def sanity_checking_with_arguments(args):
    try:
        assert args.embedding_dim > 0
    except AssertionError:
        print(f'embedding_dim must be strictly positive. Currently:{args.embedding_dim}')
        raise

    if not (args.sub_dataset_path == None or args.prop == None or bool(args.cmp_dataset)==False ):
        print(f'Invalid arguments, please specify the type of distribution you want => simple_split/property_split/complete_dataset.')
        exit(1)


    try:
        if args.eval_dataset == "FactBench":
            if not (args.sub_dataset_path == None ):
                directories = ['/data/train/domain','/data/train/domainrange','/data/train/mix','/data/train/property',
                               '/data/train/random', '/data/train/range','/data/test/domain','/data/test/domainrange',
                               '/data/test/mix','/data/test/property','/data/test/random','/data/test/range']
                for dirr in directories:
                    assert os.path.isdir(args.path_dataset_folder+dirr)
                    try:
                        assert os.path.isfile(args.path_dataset_folder +dirr+ '/'+dirr.split('/')[2]+'.txt')
                    except AssertionError:
                        print(f'The directory {args.path_dataset_folder} must contain a **'+ dirr.split('/')[2] + '.txt** .')
                        raise
            elif not (args.prop == None):
                directories = ['/properties_split/train/author','/properties_split/train/award',
                               '/properties_split/train/birthPlace','/properties_split/train/deathPlace',
                               '/properties_split/train/foundationPlace', '/properties_split/train/spouse',
                               '/properties_split/train/starring', '/properties_split/train/subsidiary',
                               '/properties_split/test/author','/properties_split/test/award',
                               '/properties_split/test/birthPlace','/properties_split/test/deathPlace',
                               '/properties_split/test/foundationPlace','/properties_split/test/spouse',
                               '/properties_split/test/starring','/properties_split/test/subsidiary']
                for dirr in directories:
                    assert os.path.isdir(args.path_dataset_folder+dirr)
                    try:
                        assert os.path.isfile(args.path_dataset_folder +dirr + '/'+ dirr.split('/')[2] + '.txt')
                    except AssertionError:
                        print(f'The directory {args.path_dataset_folder} must contain a **'+ dirr.split('/')[2] + '.txt** .')
                        raise

            elif not (bool(args.cmp_dataset)==False):
                assert os.path.isdir(args.path_dataset_folder + '/complete_dataset')
                try:
                    assert os.path.isfile(args.path_dataset_folder + '/complete_dataset/train.txt')
                    assert os.path.isfile(args.path_dataset_folder + '/complete_dataset/test.txt')
                except AssertionError:
                    print(f'The directory {args.path_dataset_folder} must contain a **train.txt** .')
                    raise
        else:
            assert os.path.isdir(args.path_dataset_folder)


    except AssertionError:
        print(f'The path does not direct to a file/folder {args.path_train_dataset}')
        raise
    # check for files here
    # try:
    #     assert os.path.isfile(args.path_dataset_folder + '/train.txt')
    # except AssertionError:
    #     print(f'The directory {args.path_dataset_folder} must contain a **train.txt** .')
    #     raise

    # args.eval = bool(args.eval)
    # args.large_kg_parse = bool(args.large_kg_parse)

def create_logger(*, name, p):
    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(p + '/info.log')
    fh.setLevel(logging.INFO)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

def select_model(args) -> Tuple[pl.LightningModule, AnyStr]:
    if str(args.model).lower() == 'temporal-model':
        model = TemporalModel(args=args)
        form_of_labelling = 'FactChecking'


    else:
        raise ValueError
    return model, form_of_labelling
