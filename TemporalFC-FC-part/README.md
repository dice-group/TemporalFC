# TemporalFC: A Temporal Fact-Checking approach for Knowledge Graphs (fact-checking component)

This project consists of 2 independent components:
1. [Fact-checking](https://github.com/dice-group/TemporalFC/TemporalFC-FC-part) component.
2. [Time-point prediction](https://github.com/dice-group/TemporalFC) component.


## Fact-checking component

This open-source project contains the Python implementation of fact checking component our approach [TemporalFC](https://github.com/factcheckerr/TemporalFC). This project is designed to ease real-world applications of fact-checking over knowledge graphs and produce better results. With this aim, we rely on:

1. [PytorchLightning](https://www.pytorchlightning.ai/) to perform training via multi-CPUs, GPUs, TPUs or  computing cluster, 
2. [Pre-trained-KG-embeddings](https://embeddings.cc/) to get pre-trained KG embeddings for knowledge graphs for knowledge graph-based component, 
3. [Elastic-search](https://www.elastic.co/blog/loading-wikipedia) to load text corpus (wikipedia) on elastic search for text-based component, and
4. [Path-based-approach](https://github.com/dice-group/COPAAL/tree/develop) to calculate output score for the path-based component.
5. [Miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers) for creating a virtual envirnament. 


## Installation
First clone the repository:
``` html
git clone https://github.com/factcheckerr/TemporalFC-FC-part.git

cd TemporalFC-FC-part
``` 

## Dataset preparation
There are two options to reproduce the results. (1) using pre-generated data, and (2) Regenerate data from scratch.
Please choose any 1 of the these 2 options.

### 1) Reusing pre-generated data
download and unzip data and embeddings files in the root folder of the project.

``` html
pip install gdown

wget https://zenodo.org/record/7913193/files/dataset.zip

wget https://zenodo.org/record/7913193/files/Embeddings.zip

wget https://zenodo.org/record/7913193/files/generated_predictions_and_models.zip


unzip dataset.zip

unzip Embeddings.zip

unzip generated_predictions_and_models.zip
``` 


Note: if it gives permission denied error you can try running the commands with "sudo"

### 2) Regenerate data from scratch (e.g., Elastic Search's output)
In case you don't want to use pre-generated data for fact-checking component, follow these steps:

For evidence sentence generation module run FactCheck on [DBpedia34k dataset](https://link.springer.com/chapter/10.1007/978-3-031-06981-9_15) and [Yago3k dataset](https://link.springer.com/chapter/10.1007/978-3-031-06981-9_15) using [Wikipedia](https://www.elastic.co/blog/loading-wikipedia) as a reference corpus. 

As an input user needs output of [FactCheck](https://github.com/dice-group/FactCheck/tree/develop-for-FROCKG-branch) in json format.

Put the result json file in the "dataset/[dataset_name]" folder.

Further details are in readme file in [overall_process folder](https://github.com/factcheckerr/TemporalFC-FC-part/tree/master/overall_process).

## Reproducing Results
Install dependencies via conda:
``` html

#setting up environment
#creating and activating conda environment

conda env create -f environment.yml

conda activate tfc

# If conda command not found: download miniconda from (https://docs.conda.io/en/latest/miniconda.html#linux-installers) and set the path: 
# export PATH=/path-to-conda/miniconda3/bin:$PATH

```
### Start generating results:

For time-based negative triples.
``` html

# Start training process, with required number of hyperparemeters. Details about other hyperparameters is in main.py file.
python main.py --emb_type Dihedron --model temporal --num_workers 32 --min_num_epochs 10 --max_num_epochs 1000 --check_val_every_n_epochs 10 --eval_dataset Dbpedia5 --negative_triple_generation corrupted-time-based

# computing evaluation files from saved model in "dataset/Hybrid_Stroage" directory
python evaluate_checkpoint_model.py --emb_type Dihedron --model temporal --num_workers 32 --min_num_epochs 10 --max_num_epochs 1000 --check_val_every_n_epochs 10 --eval_dataset Dbpedia5 --negative_triple_generation corrupted-time-based

``` 

For assertion-based negative triples.
``` html

# Start training process, with required number of hyperparemeters. Details about other hyperparameters is in main.py file.
python main.py --emb_type Dihedron --model temporal --num_workers 32 --min_num_epochs 10 --max_num_epochs 1000 --check_val_every_n_epochs 10 --eval_dataset Dbpedia5 --negative_triple_generation corrupted-triple-based

# computing evaluation files from saved model in "dataset/Hybrid_Stroage" directory
python evaluate_checkpoint_model.py --emb_type Dihedron --model temporal --num_workers 32 --min_num_epochs 10 --max_num_epochs 1000 --check_val_every_n_epochs 10 --eval_dataset Dbpedia5 --negative_triple_generation corrupted-triple-based

``` 


The prediction along with the ground truth files will be saved in "dataset/Hybrid_Stroage" directory. 

Checkpoint model will be deleted to save space but the hyperparameters to recreate the model files are also saved in "dataset/Hybrid_Stroage" directory.
##### comments:
for differnt embeddings type(emb_type) or model type(model), you just need to change the parameters.

Available temporal embeddings types:
[Dihedron](https://arxiv.org/pdf/2008.03130.pdf),

Available non-temporal embeddings types:
[TransE](https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:cr_paper_nips13.pdf), [ComplEx](https://arxiv.org/abs/2008.03130), [RDF2Vec](https://madoc.bib.uni-mannheim.de/41307/1/Ristoski_RDF2Vec.pdf) (only for BPDP dataset), [QMult](https://arxiv.org/pdf/2106.15230.pdf).

if you use graphics card and cuda issue with pytorch following command is needed in extra to execute seperately

``` html
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
``` 


Available models:
temporal, full-Hybrid

Complete available list of argument parameters can be seen in this function: [parameters list](https://github.com/factcheckerr/TemporalFC-FC-part/blob/facf5c4a352c99952642e460c2262aa50a5bfabf/main.py#L10)

Note: model names are case-sensitive. So please use exact names.

## ReGenerate AUROC results:
After computing evaluation results, the prediction files along with the model configurations and hyperparameters are saved in the "dataset/HYBRID_Storage" folder along with ground truth files.
These files can be uploaded to a live instance of [GERBIL](http://swc2017.aksw.org/gerbil/config) (by Roder et al.) framework to produce AUROC curve scores.  

## Pre-Generated AUROC results
To view the pre-generated AUROC scores online, following are the links.

#### For assertion-based negative triples:

DBpedia34k-Train: http://w3id.org/gerbil/kbc/experiment?id=202301180123

DBpedia34k-Test: http://w3id.org/gerbil/kbc/experiment?id=202301180125

Yago3k-Train: http://w3id.org/gerbil/kbc/experiment?id=202301180056

Yago3k-Test: http://w3id.org/gerbil/kbc/experiment?id=202301180129

#### For time-based negative triples:

DBpedia34k-Train: http://w3id.org/gerbil/kbc/experiment?id=202305020014

DBpedia34k-Test: http://w3id.org/gerbil/kbc/experiment?id=202305020015

Yago3k-Train: http://w3id.org/gerbil/kbc/experiment?id=202305020013

Yago3k-Test: http://w3id.org/gerbil/kbc/experiment?id=202305020012

Note: Prediction files can be viewed in results folder

## Future plan:
As future work, we will exploit the modularity of TemporalFC by integrating rule-based approaches. We also plan to explore other possibilities to select the best evidence sentences.

## Acknowledgement 
To be added later!
## Authors
To be added later!






