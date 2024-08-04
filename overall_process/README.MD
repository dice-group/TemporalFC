
### Pre-requisite
1) Put textual evidence results from any search engine in  [dataset_name]_factcheckoutput.txt file and put it in "dataset" directory. Note: This result file can be obtained from [FactCheck](https://github.com/dice-group/FactCheck/tree/develop-for-FROCKG-branch) project.
3) Format of the file should be as following:


``` html
-=-=-=-=-=-=-==-=-==-=-=-=-==-=-=-=-=-=-=-==-=-=-=-
10
/factbench/test/correct/death/death_00053.ttl
 defactoScore: 0.98 setProofSentences : [ComplexProofs{website='https://en.wikipedia.org/wiki/Reba White Williams', proofPhrase='In 1999 , White Williams ran unsuccessfully for the New York City City Council in District 4 .', trustworthinessScore='0.997908778988452'}, ComplexProofs{website='https://en.wikipedia.org/wiki/James Leo Herlihy', proofPhrase='Like Williams , Herlihy had lived in New York City .', trustworthinessScore='0.9975670565782072'}, ComplexProofs{website='https://en.wikipedia.org/wiki/Charles Williams (musician)', proofPhrase='Charles Isaac Williams -LRB- born July 18 , 1932 -RRB- is an alto saxophonist based in New York City .', trustworthinessScore='0.9991775993927828'}] subject : Tennessee Williams object : New York City predicate deathPlace
-=-=-=-=-=-=-==-=-==-=-=-=-==-=-=-=-=-=-=-==-=-=-=-

``` 
Note: all entries are not required. Only proofPhrase is used. 
### Steps
1) Run GenerateTrainTestTripleSet.py file to generate Train and Test triple from FactCheck output file.
2) Run sentenceEmbeddings.py file to generate Train and Test sentence embeddings from FactCheck output file.
3) For full-hybrid approach, put the COPAAL result in "dataset/copaal" folder and run GenerateTrainTestTripleSetFullHybrid.py and sentenceEmbeddingsFullHybrid.py files.

#### further pre-req
4) arrange first true statements and then false.