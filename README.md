# TemporalFC: A Temporal Fact-Checking Approach for Knowledge Graphs
<p><img src = "https://files.dice-research.org/datasets/ISWC2023_TemporalFC//logo.jpeg" alt = "TemporalFC Logo" width = "30%" align = "center"></p>

This open-source project contains the Python implementation of our approach [TemporalFC](https://papers.dice-research.org/2023/ISWC_TemporalFC/public.pdf) (published at ISWC2023). This project is designed to ease real-world applications of fact-checking over knowledge graphs and produce better results. With this aim, we rely on:

1. [PytorchLightning](https://www.pytorchlightning.ai/) to perform training via multi-CPUs, GPUs, TPUs or  computing cluster, 
2. [Pre-trained-TKG-embeddings](https://link.springer.com/chapter/10.1007/978-3-031-06981-9_15) to get pre-trained TKG embeddings for knowledge graphs for knowledge graph-based component, 
3. [Elastic-search](https://www.elastic.co/blog/loading-wikipedia) to load text corpus (wikipedia) on elastic search for text-based component, and
4. [Path-based-approach](https://github.com/dice-group/COPAAL/tree/develop) to calculate output score for the path-based component.

This project performs 2 independent tasks:
1. Fact-checking
2. Time-point prediction


## Installation
First clone the repository:
``` html
git clone https://github.com/dice-group/TemporalFC.git

cd TemporalFC
``` 

## Reproducing Results
There are two options to reproduce results. 
1) Using pre-processed input dataset, and 
2) Regenerate input dataset from scratch.

Select any 1 of these 2 options.

### 1) Re-Using pre-generated dataset
download and unzip data and embeddings files in the root folder of the project.

``` html
pip install gdown

wget https://files.dice-research.org/datasets/ISWC2023_TemporalFC/data_TP.zip

unzip data_TP.zip
``` 

Note: if it gives permission denied error you can try running the commands with "sudo"



### 2) Generating dataset from scratch
To regenerate data from scratch, you need to re-train the embedding algorithm again and put the generated embeddings in data_TP/dataset_name/embeddings folder, and dataset in data_TP/dataset_name/train and data_TP/dataset_name/test foder.

Detailed instructions are in [overall_process](https://github.com/dice-group/TemporalFC/tree/main/overall_process) folder.

## Running experiments
Install dependencies via conda:
``` html

#setting up environment
#creating and activating conda environment

conda env create -f environment.yml

conda activate tfc

#If conda command not found: download miniconda from (https://docs.conda.io/en/latest/miniconda.html#linux-installers) and set the path: 
#export PATH=/path-to-conda/miniconda3/bin:$PATH

```
start generating results:

#### Fact Checking component
``` html

# Start training process, with required number of hyperparemeters. Details about other hyperparameters is in main.py file.
python main.py --eval_dataset Dbpedia124k --model temporal-full-hybrid  --max_num_epochs 500   --min_num_epochs 50 --batch_size 12000 --val_batch_size 1000  --negative_triple_generation corrupted-triple-based  --task fact-checking --emb_type dihedron --embedding_dim 100 --num_workers 1
# computing evaluation files from saved model in "dataset/Hybrid_Stroage" directory
python evaluate_checkpoint_model_FC.py --checkpoint_dir_folder all --checkpoint_dataset_folder dataset/  --eval_dataset Dbpedia124k --model temporal-full-hybrid  --max_num_epochs 500   --min_num_epochs 50 --batch_size 12000 --val_batch_size 1000  --negative_triple_generation corrupted-triple-based  --task fact-checking --emb_type dihedron --embedding_dim 100 --num_workers 1

``` 

#### Time-point prediction component

``` html

# Start training process, with required number of hyperparemeters. Details about other hyperparameters is in main.py file.
python main.py --eval_dataset Dbpedia124k --model temporal-prediction-model  --max_num_epochs 500   --min_num_epochs 50 --batch_size 12000 --val_batch_size 1000  --negative_triple_generation False  --task time-prediction --emb_type dihedron --embedding_dim 100 --num_workers 1
# computing evaluation files from saved model in "dataset/Hybrid_Stroage" directory
python evaluate_checkpoint_model_TP.py --checkpoint_dir_folder all --checkpoint_dataset_folder dataset/  --eval_dataset Dbpedia124k --model temporal-prediction-model  --max_num_epochs 500   --min_num_epochs 50 --batch_size 12000 --val_batch_size 1000  --negative_triple_generation False  --task time-prediction --emb_type dihedron --embedding_dim 100 --num_workers 1

``` 

##### comments:
1. To reproduce exact results you have to use exact parameters as listed above.

2. For other datasets you need to change the parameter in front of --eval_dataset

3. Use parallel processing for fast processing. Default parameter is set to 4 workers that we used to generate results.

Available embeddings types:
[dihedron](https://link.springer.com/chapter/10.1007/978-3-031-06981-9_15)

Available models:
temporal-prediction-model, temporal-full-hybrid

Note: model names are case-sensitive. So please use exact names.

#### Fact checking part:
Fact checking part should contain negative triple generation parameter. 

Available options are: (1) corrupted-triple-based and (2) corrupted-time-based,

## Future plan:
As future work, we will exploit the modularity of TemporalFC by integrating time-period based fact checking. 

## Acknowledgement 
The work has been supported by the EU H2020 Marie Skłodowska-Curie project KnowGraphs (no. 860801)).

## References
If you find our work useful in your research, please consider citing the respective paper:

```bibtex

#TemporalFC
@inproceedings{qudus2023TemporalFC,
author = {Qudus, Umair and R\"{o}der, Michael and Kirrane, Sabrina and Ngomo, Axel-Cyrille Ngonga},
title = {TemporalFC: A Temporal Fact Checking Approach over&nbsp;Knowledge Graphs},
year = {2023},
isbn = {978-3-031-47239-8},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-3-031-47240-4_25},
doi = {10.1007/978-3-031-47240-4_25},
booktitle = {The Semantic Web – ISWC 2023: 22nd International Semantic Web Conference, Athens, Greece, November 6–10, 2023, Proceedings, Part I},
pages = {465–483},
numpages = {19},
keywords = {temporal fact checking, ensemble learning, transfer learning, time-point prediction, temporal knowledge graphs},
location = {Athens, Greece}
}


#HybridFC

@InProceedings{qudus2022HybridFC,
  Author      = {Qudus, Umair and Röder, Michael and Saleem,Muhammad and Ngomo, Axel-Cyrille Ngonga},
  Editor       ={Sattler, Ulrike and Hogan, Aidan and Keet, Maria and Presutti, Valentina and Almeida, Jo{\~a}o Paulo A. and Takeda, Hideaki and Monnin, Pierre and Pirr{\`o}, Giuseppe and d'Amato, Claudia},
  Title          = {HybridFC: A Hybrid Fact-Checking Approach for Knowledge Graphs},
  booktitle   = {The Semantic Web -- ISWC 2022},
  Year          = {2022},
  Doi            = {10.1007/978-3-031-19433-7\_27},
  isbn           ={978-3-031-19433-7},
  pages        = {462--480},
  address     ={Cham},
  publisher   =  {Springer International Publishing},
  biburl        = {https://www.bibsonomy.org/bibtex/2ec2f0b9ee7ca0c1c6ef1d8fbcd7262e4/dice-research},
  keywords  = {knowgraphs frockg raki 3dfed dice ngonga saleem roeder qudus},
  url             = {https://papers.dice-research.org/2022/ISWC_HybridFC/public.pdf},
}


```

In case you have any question, please contact: ```umair.qudus@uni-paderborn.de``` or ```umair.qudus@hotmail.com```

## Authors
* [Umair Qudus](https://dice-research.org/UmairQudus) (DICE, Paderborn University) 
* [ Michael Röder](https://dice-research.org/MichaelRoeder) (DICE,  Paderborn University) 
* [Sabrina Kirrane](http://sabrinakirrane.com/) (WU,  WU Vienna) 
* [Axel-Cyrille Ngonga Ngomo](https://dice-research.org/AxelCyrilleNgongaNgomo) (DICE,  Paderborn University)
  






