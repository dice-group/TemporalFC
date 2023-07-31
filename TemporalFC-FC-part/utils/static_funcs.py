import os
import logging
import datetime
from scipy import stats
import matplotlib.pyplot as plt

from nn_models.path_KGE_hybrid_model import PathKGEHybridModel
from nn_models.path_model import PathModel
from nn_models.temporal_model import TemporalModel
from nn_models.text_KGE_hybrid_model import TextKGEHybridModel
from nn_models.complex import ConEx
from nn_models.KGE_model import KGEModel
from nn_models.full_hybrid_model import FullHybridModel
import pytorch_lightning as pl
from typing import AnyStr, Tuple

from nn_models.text_model import TextModel
from nn_models.text_path_hybrid_model import TextPathHybridModel


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
    # arg.checkpoint_callback = False
    arg.enable_checkpointing = True
    arg.find_unused_parameters = False
    arg.logger = True
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

    if not (args.sub_dataset_path == None or args.prop_split == None or bool(args.cmp_dataset)==False ):
        print(f'Invalid arguments, please specify the type of distribution you want => simple_split/property_split/complete_dataset.')
        exit(1)
    try:
        if str(args.model).lower() == "temporal":
            assert os.path.isfile("Embeddings/" + args.emb_type +"/"+str(args.eval_dataset).lower()+ "/all_times_embeddings.txt")
    except AssertionError:
        print(f'Temporal embeddings missing from :{"Embeddings/" + args.emb_type +"/"+str(args.eval_dataset).lower()} folder')
        raise

    if args.model !="text-only":
        try:
            assert os.path.isfile("Embeddings/" + args.emb_type +"/"+str(args.eval_dataset).lower()+ "/all_entities_embeddings.txt")
            assert os.path.isfile("Embeddings/" + args.emb_type +"/"+str(args.eval_dataset).lower()+ "/all_relations_embeddings.txt")
        except AssertionError:
            print(f'Embeddings missing from :{"Embeddings/" + args.emb_type +"/"+str(args.eval_dataset).lower()} folder')
            raise

    try:
        if str(args.eval_dataset).lower() == "dbpedia5":
            # assert args.path_dataset_folder == "dataset/complete_dataset/dbpedia34k/"
            assert args.path_dataset_folder == "dataset/"
            try:
                assert os.path.isdir(args.path_dataset_folder + str(args.eval_dataset).lower())
            except AssertionError:
                print(f'The directory {args.path_dataset_folder} must contain ' + str(args.eval_dataset).lower() )
                raise


        if str(args.eval_dataset).lower() == "factbench":
            if not (args.sub_dataset_path == None ):
                directories = ['/train/domain','/train/domainrange','/train/mix','/train/property',
                               '/train/random','/test/domain','/test/domainrange',
                               '/test/mix','/test/property','/test/random','/test/range', '/train/range'] #,'/hybrid_data/test/range', '/hybrid_data/train/range'
                for dirr in directories:
                    assert os.path.isdir(args.path_dataset_folder+str(args.eval_dataset).lower()+dirr)
                    try:
                        assert os.path.isfile(args.path_dataset_folder +str(args.eval_dataset).lower()+dirr+ '/'+dirr.split('/')[1]+'.txt')
                    except AssertionError:
                        print(f'The directory {args.path_dataset_folder} must contain a **'+ dirr.split('/')[1] + '.txt** .')
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
                    assert os.path.isdir(args.path_dataset_folder+str(args.eval_dataset).lower()+dirr)
                    try:
                        assert os.path.isfile(args.path_dataset_folder +str(args.eval_dataset).lower()+dirr + '/'+ dirr.split('/')[2] + '.txt')
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
                print(f'Please specify sub dataset path.')

        else:
            assert os.path.isdir(args.path_dataset_folder)


    except AssertionError:
        print(f'The path does not direct to a file/folder {args.path_dataset_folder}')
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
    if str(args.model).lower() == 'text-kge-hybrid':
        model = TextKGEHybridModel(args=args)
        form_of_labelling = 'TriplePrediction'

    elif str(args.model).lower() == 'path-only':
        model = PathModel(args=args)
        form_of_labelling = 'TriplePrediction'

    elif str(args.model).lower() == 'text-only':
        model = TextModel(args=args)
        form_of_labelling = 'TriplePrediction'

    elif str(args.model).lower() == 'kge-only':
        model = KGEModel(args=args)
        form_of_labelling = 'TriplePrediction'

    elif str(args.model).lower() == 'full-hybrid':
        model = FullHybridModel(args=args)
        form_of_labelling = 'TriplePrediction'
    elif str(args.model).lower() == 'temporal':
        model = TemporalModel(args=args)
        form_of_labelling = 'TriplePrediction'

    elif str(args.model).lower() == 'kge-path-hybrid':
        model = PathKGEHybridModel(args=args)
        form_of_labelling = 'TriplePrediction'

    elif str(args.model).lower() == 'path-kge-hybrid':
        model = PathKGEHybridModel(args=args)
        form_of_labelling = 'TriplePrediction'

    elif str(args.model).lower() == 'text-path-hybrid':
        model = TextPathHybridModel(args=args)
        form_of_labelling = 'TriplePrediction'

    elif str(args.model).lower() == 'path-text-hybrid':
        model = TextPathHybridModel(args=args)
        form_of_labelling = 'TriplePrediction'

    else:
        raise ValueError
    return model, form_of_labelling
