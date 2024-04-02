from executer_TP import Execute_TP
import pytorch_lightning as pl
import argparse
import os
from pytorch_lightning import Trainer, seed_everything
seed_everything(42, workers=True)

current_dir = os.getcwd()
DATA_PATH = os.path.join(current_dir,"data_TP")

def argparse_default(description=None):
    parser = pl.Trainer.add_argparse_args(argparse.ArgumentParser())
    # Paths.
    parser.add_argument("--path_dataset_folder", type=str, default='data_TP/')

    parser.add_argument("--storage_path", type=str, default='HYBRID_Storage')
    parser.add_argument("--eval_dataset", type=str, default='Dbpedia124k',
                        help="Available datasets: Dbpedia124k, Yago3K, Wikidata")
    # FactBench, BPDP,Dbpedia34k,
    parser.add_argument("--sub_dataset_path", type=str, default=None,
                        help="Available subpaths: bpdp/, domain/, domainrange/, mix/, property/, random/, range/,")
    parser.add_argument("--prop", type=str, default=None,
                        help="Available properties (only for FactBench dataset if available): architect, artist, author, commander, director, musicComposer, producer, None")
    parser.add_argument("--negative_triple_generation", type=str, default="corrupted-triple-based",
                        help="Available approaches: corrupted-triple-based, corrupted-time-based, False")
    parser.add_argument("--cmp_dataset", type=bool, default=True)
    parser.add_argument("--include_veracity", type=bool, default=True)

    # parser.add_argument("--auto_scale_batch_size", type=bool, default=True)
    # parser.add_argument("--deserialize_flag", type=str, default=None, help='Path of a folder for deserialization.')

    # Models. select temporal model for time point prediction!!
    parser.add_argument("--model", type=str, default='hybridfc-full-Hybrid',
                        help="Available models:temporal-prediction-model, temporal-full-hybrid, hybridfc-full-Hybrid, KGE-only,text-only, text-KGE-Hybrid, path-only, text-path-Hybrid, KGE-path-Hybrid")


    parser.add_argument("--task", type=str, default='fact-checking',
                        help="Available datasets:   time-prediction, fact-checking")
                        # help="Available models:Hybrid, ConEx, TransE, Hybrid, ComplEx, RDF2Vec")

    parser.add_argument("--emb_type", type=str, default='CoNeX',
                        help="Available TKG embeddings: dihedron, torchbiggraph, TransE, CoNeX, None")

    # Hyperparameters pertaining to number of parameters.
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--valid_ratio', type=int, default=20)
    parser.add_argument('--sentence_dim', type=int, default=768)
    parser.add_argument("--max_num_epochs", type=int, default=50)
    parser.add_argument("--min_num_epochs", type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--val_batch_size', type=int, default=5)
    # parser.add_argument('--negative_sample_ratio', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of cpus used during batching')
    parser.add_argument("--check_val_every_n_epochs", type=int, default=10)
    # parser.add_argument('--enable_checkpointing', type=bool, default=True)
    # parser.add_argument('--deterministic', type=bool, default=True)
    # parser.add_argument('--fast_dev_run', type=bool, default=False)
    # parser.add_argument("--accumulate_grad_batches", type=int, default=3)
    # PREPROCESS DATASETS
    parser.add_argument("--preprocess", type=str, default='False',
                        help="Available options: False, Concat, SentEmb, TrainTestTriplesCreate")

    parser.add_argument("--ids_only", type=str, default=False)

    if description is None:
        return parser.parse_args()
    else:
        return parser.parse_args(description)

if __name__ == '__main__':
    args = argparse_default()
    exc = Execute_TP(args)
    exc.start()







    # Yago al
    # if args.eval_dataset == "Yago3K":
    #     args.ids_only = True

    # Preprocess dataset if flag is True!
    # if args.preprocess != 'False':
    #     if args.preprocess == 'Concat':
    #         if args.eval_dataset == "Dbpedia124k":
    #             DBpedia34kDataset = True
    #             # ConcatEmbeddings(args, path_dataset_folder=args.path_dataset_folder,DBpedia34k=DBpedia34kDataset)
    #         print("concat done")
    #     elif args.preprocess == 'SentEmb':
    #         print("sentence vectors creation is done")
    #     elif args.preprocess == 'TrainTestTriplesCreation':
    #         print("Train and Test triples creation is complete")

    # if args.eval_dataset == "Dbpedia124k" or args.eval_dataset == "Yago3K":
    # else:
    #     print("Please specify a valid dataset")
    #     exit(1)
