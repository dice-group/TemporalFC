import os
import logging
import datetime
from scipy import stats
import matplotlib.pyplot as plt
import string

from nn_models_TP.KGE_model import KGEModel
from nn_models_TP.temporal_full_hybrid_model import TemporalFullHybridModel
from nn_models_TP.hybrid_fc_model import HybridFCModel
from nn_models_TP.path_KGE_hybrid_model import PathKGEHybridModel
from nn_models_TP.path_model import PathModel
from nn_models_TP.temporal_prediction_model import TemporalPredictionModel

import pytorch_lightning as pl
from typing import AnyStr, Tuple

from nn_models_TP.text_KGE_hybrid_model import TextKGEHybridModel
from nn_models_TP.text_model import TextModel
from nn_models_TP.text_path_hybrid_model import TextPathHybridModel


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
# Function to check if a string contains only numeric characters (potential ID)
def is_numeric_string(s):
    return s.isdigit()
def remove_punctuations(s):
    tranlator = str.maketrans("","",string.punctuation)
    result = s.translate(tranlator)
    return result
# Function to check if a string contains alphabetic characters (potential string)
def is_alphabetic_string(s):
    if s.startswith("<") and s.endswith(">"):
        s = remove_punctuations(s) #s[1:-1].replace(":","").replace("/","").replace(".","").replace(",","").replace("_","").replace("-","")
    return s.isalpha()

# Function to analyze the file
def is_numeric(file_path):
    try:
        assert os.path.isfile(file_path)
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line into words (assuming space-separated words)
                words = line.strip().split()
                for word in words:
                    if is_numeric_string(word):
                        return True
                        # print(f'Found a potential ID: {word}')
                    elif is_alphabetic_string(word):
                        return False
                        #print(f'Found a potential string: {word}')
                    else:
                        print(f'Unrecognized: {word}')
    except FileNotFoundError:
        print(f'File not found: {file_path}')

def sanity_checking_with_arguments(args):
    try:
        assert args.embedding_dim > 0
    except AssertionError:
        print(f'embedding_dim must be strictly positive. Currently:{args.embedding_dim}')
        raise

    if not (args.sub_dataset_path == None or args.prop == None or bool(args.complete_dataset)==False ):
        print(f'Invalid arguments, please specify the type of distribution you want => simple_split/property_split/complete_dataset.')
        exit(1)

    try:
        print("test")
        # assert os.path.isdir(args.path_dataset_folder + str(args.eval_dataset).lower() + '/embeddings/' +args.emb_type)
        # assert os.path.isdir(args.path_dataset_folder + str(args.eval_dataset).lower() + '/embeddings/' +args.emb_type)
    except AssertionError:
        print('The directory' + str(args.path_dataset_folder) + str(args.eval_dataset).lower() + '/embeddings/'  +str(args.emb_type)  + 'must contain a **embedding files (entity_embeddings.csv and relation_embeddings.csv)** .')
        raise
    try:
        if str(args.task).lower() == "fact-checking":
            try:
                assert args.model != "temporal-prediction-model"
            except AssertionError:
                print(f'For fact-checking task you can NOT chose the temporal-prediction-model!')
                raise
            try:
                assert args.negative_triple_generation != "False"
            except AssertionError:
                print(f'For fact-checkingl task you must specify a negative triple generation method!')
                raise
        elif str(args.task).lower() == "time-prediction":
            try:
                assert args.model == "temporal-prediction-model"
            except AssertionError:
                print(f'For time-prediction task you can chose the temporal-prediction-model only!!')
                raise

        else:
            assert False

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
                assert os.path.isdir(args.path_dataset_folder + str(args.eval_dataset).lower()+"/")
                try:
                    assert os.path.isfile(args.path_dataset_folder + str(args.eval_dataset).lower() +'/train/train')
                    assert os.path.isfile(args.path_dataset_folder + str(args.eval_dataset).lower() +'/test/test')
                except AssertionError:
                    print(f'The directory {args.path_dataset_folder} must contain a **train.txt** .')
                    raise
        else:
            try:
                assert os.path.isdir(args.path_dataset_folder+str(args.eval_dataset).lower()+"/")
                if args.ids_only == True:
                    if is_numeric(args.path_dataset_folder + str(args.eval_dataset).lower() + "/train/train") == False:
                        print("Numeric IDs not found, Please change the flag! \"ids_only\" to False")
                        raise
                else:
                    if is_numeric(args.path_dataset_folder + str(args.eval_dataset).lower() + "/train/train") == True:
                        print("Numeric IDs found, Please change the flag! \"ids_only\" to True")
                        raise


            except AssertionError:
                print(f'{args.path_dataset_folder+str(args.eval_dataset).lower()+"/"} should be valid path .')
                raise



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
    if str(args.model).lower() == 'temporal-prediction-model':
        model = TemporalPredictionModel(args=args)
        form_of_labelling = 'TimePrediction'
    #     other fact checking models (#tobeupdated later)
    elif str(args.model).lower() == 'temporal-full-hybrid':
        form_of_labelling = 'FactChecking'
        model = TemporalFullHybridModel(args=args)
    elif str(args.model).lower() == 'hybridfc-full-hybrid':
        form_of_labelling = 'FactChecking'
        model = HybridFCModel(args=args)
    elif str(args.model).lower() == 'kge-only':
        form_of_labelling = 'FactChecking'
        model = KGEModel(args=args)
    elif str(args.model).lower() == 'text-only':
        form_of_labelling = 'FactChecking'
        model = TextModel(args=args)
    elif str(args.model).lower() == 'text-kge-hybrid':
        form_of_labelling = 'FactChecking'
        model = TextKGEHybridModel(args=args)
    elif str(args.model).lower() == 'path-only':
        form_of_labelling = 'FactChecking'
        model = PathModel(args=args)
    elif str(args.model).lower() == 'text-path-hybrid':
        form_of_labelling = 'FactChecking'
        model = TextPathHybridModel(args=args)
    elif str(args.model).lower() == 'kge-path-hybrid':
        form_of_labelling = 'FactChecking'
        model = PathKGEHybridModel(args=args)

    else:
        raise ValueError
    return model, form_of_labelling
