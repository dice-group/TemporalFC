# TemporalFC: A Temporal Fact-Checking Approach for Knowledge Graphs
## Time predication component

This open-source project contains the Python implementation of time predication component our approach [TemporalFC](https://github.com/factcheckerr/TemporalFC). This project is designed to ease real-world applications of fact-checking over knowledge graphs and produce better results. With this aim, we rely on:

1. [PytorchLightning](https://www.pytorchlightning.ai/) to perform training via multi-CPUs, GPUs, TPUs or  computing cluster, 
2. [Pre-trained-TKG-embeddings](https://link.springer.com/chapter/10.1007/978-3-031-06981-9_15) to get pre-trained TKG embeddings for knowledge graphs for knowledge graph-based component, 
3. [Elastic-search](https://www.elastic.co/blog/loading-wikipedia) to load text corpus (wikipedia) on elastic search for text-based component, and
4. [Path-based-approach](https://github.com/dice-group/COPAAL/tree/develop) to calculate output score for the path-based component.

This project consists of 2 independent components:
1. [Fact-checking](https://github.com/dice-group/TemporalFC/tree/e1b92967b0a2608c2e089bd1a13f59420c491f9b/TemporalFC-FC-part) component.
2. [Time-point prediction](https://github.com/dice-group/TemporalFC) component.


## Installation
First clone the repository:
``` html
git clone https://github.com/factcheckerr/TemporalFC.git

cd TemporalFC
``` 

## Reproducing Results
There are two options to repreoduce the results. (1) using pre-generated data, and (2) Regenerate data from scratch.
Please chose any 1 of the these 2 options.

### 1) Re-Using pre-generated data
download and unzip data and embeddings files in the root folder of the project.

``` html
pip install gdown

wget https://zenodo.org/records/8197242/files/data_TP.zip
wget https://zenodo.org/records/7552968/files/dataset.zip
wget https://zenodo.org/records/7552968/files/Embeddings.zip

unzip data.zip
``` 

Note: if it gives permission denied error you can try running the commands with "sudo"



### 2) Generating data from scratch
To regenerate data from scratch, you need to run embedding algorithm again and put the generated SPO embedding in data/dataset_name/temporal_embeddings folder, and dataset in data/dataset_name/train and data/dataset_name/test foder.


## Running experiments
Install dependencies via conda:
``` html

#setting up environment
#creating and activating conda environment

conda env create -f environment.yml

conda activate hfc2

#If conda command not found: download miniconda from (https://docs.conda.io/en/latest/miniconda.html#linux-installers) and set the path: 
#export PATH=/path-to-conda/miniconda3/bin:$PATH

```
start generating results:
``` html

# Start training process, with required number of hyperparemeters. Details about other hyperparameters is in main.py file.
python run.py             --dataset DBPedia5             --model sFourDETim             --rank 100             --regularizer N3             --reg 0.00000000001             --optimizer Adagrad             --max_epochs 500             --patience 15             --valid 10             --batch_size 100             --neg_sample_size -1             --init_size 0.001             --learning_rate 0.1             --gamma 0.0             --bias learn             --dtype single             --double_neg             --cuda_n 2             --dataset_type quintuple_Tim
# computing evaluation files from saved model in "dataset/Hybrid_Stroage" directory
python evaluate_checkpoint_model.py        --dataset DBPedia5             --model sFourDETim             --rank 100             --regularizer N3             --reg 0.00000000001             --optimizer Adagrad             --max_epochs 500             --patience 15             --valid 10             --batch_size 100             --neg_sample_size -1             --init_size 0.001             --learning_rate 0.1             --gamma 0.0             --bias learn             --dtype single             --double_neg             --cuda_n 2             --dataset_type quintuple_Tim
``` 

##### comments:
1. To reproduc exact results you have to use exact parameters as listed above.

2. For other datasets you need to change the parameter in front of --dataset

3. Use GPU for fast processing. Default parameter is set to 2 GPUs that we used to generate results.

4. For different embeddings type(emb_type) or model type(model), you just need to change the parameters. 

Available embeddings types:
[dihedron](https://link.springer.com/chapter/10.1007/978-3-031-06981-9_15), [T-TransE](https://aclanthology.org/C16-1161/), [T-ComplEx]().

Available models:
temporal-model

Note: model names are case-sensitive. So please use exact names.

## Fact checking part:
Fact checking part of TemporalFC is available in a second project: [TemporalFC-FactCheck](https://github.com/factcheckerr/TemporalFC-FC-part).  

## Future plan:
As future work, we will exploit the modularity of TemporalFC by integrating time-period based fact checking. 

## Acknowledgement 
The work has been supported by the EU H2020 Marie Skłodowska-Curie project KnowGraphs (no. 860801)).

## Authors
 * [Umair Qudus](https://dice-research.org/UmairQudus) (DICE, Paderborn University) 
 * [ Michael Röder](https://dice-research.org/MichaelRoeder) (DICE,  Paderborn University) 
 * [Sabrina Kirrane](http://sabrinakirrane.com/) (WU,  Vienna University) 
 * [Axel-Cyrille Ngonga Ngomo](https://dice-research.org/AxelCyrilleNgongaNgomo) (DICE,  Paderborn University)






