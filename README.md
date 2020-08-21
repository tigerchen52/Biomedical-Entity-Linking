# Biomedical-Entity-Linking
![model](https://ss0.bdstatic.com/70cFvHSh_Q1YnxGkpoWK1HF6hhy/it/u=702257389,1274025419&fm=27&gp=0.jpg "区块链")

## Environment setup
Clone the repository and set up the environment via "requirements.txt". Here we use python3.6. 
```
pip3 install -r requirements.txt
```
## Data preparation
**Dataset.** We valid our model on three datasets, ShARe/CLEF, NCBI and ADR. Download these dataset and their corresponding knowledge bases following the urls below.
| Dataset | Reference KB  |
|------|------|
| [NCBI disease](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/) | [MEDIC (July 6, 2012)](http://ctdbase.org/downloads/#alldiseases) |
| [ShARe/CLEF eHealth 2013 Challenge](https://physionet.org/content/shareclefehealth2013/1.0/) | [SNOMED-CT (UMLS 2012AA)](https://www.nlm.nih.gov/pubs/techbull/mj12/mj12_umls_2012aa_release.html)|
| [TAC 2017 ADR](https://bionlp.nlm.nih.gov/tac2017adversereactions/) | [MedDRA (18.1)](https://www.meddra.org/) |

**Word Embedding.** 
In our experiments, we represent each word by a 200-dimensional word embedding computed on PubMed and
MIMIC-III corpus, which is proposed in this paper[1]. [Downlaod](https://github.com/ncbi-nlp/BioSentVec)

**Extra Biomedical documents.**
entities in the same document have a co-occurrence relationship to some extent,
which can be used to enhance entity liking. To capture this relationship among entities, we adopt the
method of pre-trained entity embedding. More specifically, we treat each entity occurring in the same
document as a single word. Hence, each document can be represented as a sentence and each word in it is
an entity. Then utilizing the word2vec model to get pre-trained entity embeddings
so that entities often co-occur together have a similar distributed representation.
Here, the medical corpus we adopt is a collection of PubMed abstracts
which can be obtained at *ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/*


## Reference
[1] Zhang Y, Chen Q, Yang Z, Lin H, Lu Z. BioWordVec, improving biomedical word embeddings with subword information and MeSH. Scientific Data. 2019.
