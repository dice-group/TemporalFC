import argparse
from executer import Execute
from overall_process.ConcatenateEmbeddings import ConcatEmbeddings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

seed_everything(42, workers=True)

def argparse_default(description=None):
    parser = pl.Trainer.add_argparse_args(argparse.ArgumentParser())

    # Models.
    parser.add_argument("--model", type=str, default='temporal',
                        help="Available models:temporal, full-Hybrid, KGE-only,text-only, text-KGE-Hybrid, path-only, text-path-Hybrid, KGE-path-Hybrid")
    # help="Available models:Hybrid, ConEx, TransE, Hybrid, ComplEx, RDF2Vec")

    # Paths.
    parser.add_argument("--path_dataset_folder", type=str, default='dataset/',
                        help="dataset/") # dataset/complete_dataset/dbpedia34k/,
    parser.add_argument("--storage_path", type=str, default='HYBRID_Storage')
    # parser.add_argument("--temporal", type=str, default='HYBRID_Storage')log_every_n_steps

    # dataset config
    parser.add_argument("--eval_dataset", type=str, default='Dbpedia5',
                        help="Available datasets: Dbpedia5, Yago5, FactBench, BPDP, WTR")

    parser.add_argument("--negative_triple_generation", type=str, default="corrupted-triple-based",
                        help="Available approaches: corrupted-triple-based, corrupted-time-based, False")

    parser.add_argument("--sub_dataset_path", type=str, default=None,
                        help="Available subpaths in case of FactBench dataset: bpdp/, domain/, domainrange/, mix/, property/, random/, range/,")

    # for properties split. only works for FactBench yet #todo later
    parser.add_argument("--prop_split", type=str, default=None)
    parser.add_argument("--cmp_dataset", type=bool, default=False)



    parser.add_argument("--emb_type", type=str, default='Dihedron',
                        help="Available KG embeddings: ConEx, TransE, ComplEx, RDF2Vec, QMult,  Dihedron")

    # Hyperparameters pertaining to number of parameters.
    parser.add_argument('--embedding_dim', type=int, default=50)
    parser.add_argument('--sentence_dim', type=int, default=768)
    parser.add_argument("--max_num_epochs", type=int, default=10)
    parser.add_argument("--min_num_epochs", type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=8, help='Number of cpus used during batching')
    parser.add_argument("--check_val_every_n_epochs", type=int, default=10)

    # Dataset preprocessing for HybridFC input format formulation
    parser.add_argument("--preprocess", type=str, default='False',
                        help="Available options: False, Concat, SentEmb, TrainTestTriplesCreate")
    parser.add_argument("--valid", type=str, default='False',
                        help="Available options: True, False")


    # some configuration #todo to be added later
    # parser.add_argument('--batch_size', type=int, default=345)
    # parser.add_argument('--val_batch_size', type=int, default=345)
    # parser.add_argument('--negative_sample_ratio', type=int, default=0)
    # parser.add_argument('--enable_checkpointing', type=bool, default=True)
    # parser.add_argument('--deterministic', type=bool, default=True)
    # parser.add_argument('--fast_dev_run', type=bool, default=False)
    # parser.add_argument("--accumulate_grad_batches", type=int, default=3)
    # parser.add_argument("--auto_scale_batch_size", type=bool, default=True)
    # parser.add_argument("--deserialize_flag", type=str, default=None, help='Path of a folder for deserialization.')


    if description is None:
        return parser.parse_args()
    else:
        return parser.parse_args(description)

if __name__ == '__main__':
    args = argparse_default()
    Execute(args).start()
    exit(1)






    # available_models = ["full-Hybrid", "KGE-only", "text-only", "path-only", "text-KGE-Hybrid", "text-path-Hybrid", "KGE-path-Hybrid"]
    # max_epochs = [20,50,100,200,500]
    # for epoc in max_epochs:
        # for mdl in available_models:
        #     print("model started:::::::::::::"+mdl)

    #
    # if args.preprocess != 'False':
    #     if args.preprocess == 'Concat':
    #         if args.eval_dataset == "Dbpedia5":
    #             DBpedia34kDataset = True
    #             ConcatEmbeddings(args, path_dataset_folder=args.path_dataset_folder,DBpedia34k=DBpedia34kDataset)
    #         print("concat done")
    #     elif args.preprocess == 'SentEmb':
    #         print("sentence vectors creation is done")
    #     elif args.preprocess == 'TrainTestTriplesCreation':
    #         print("Train and Test triples creation is complete")
    #
    # elif args.eval_dataset == "FactBench":
    #     datasets_class = [ "property/","range/", "domain/", "domainrange/", "mix/", "random/"]
    #     for cls in datasets_class:
    #         args = argparse_default()
    #             # args.max_num_epochs = epoc
    #             # args.model = mdl
    #         args.subpath = cls
    #         Execute(args).start()
    #                 # exit(1)
    # elif args.eval_dataset=="BPDP":
    #     args = argparse_default()
    #     Execute(args).start()
    #             # exit(1)
    # elif args.eval_dataset == "Dbpedia5":
    #     Execute(args).start()
    # else:
    #     print("Please specify the dataset")
    #     exit(1)




