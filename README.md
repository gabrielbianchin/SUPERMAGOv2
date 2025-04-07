# SUPERMAGOv2: Protein Function Prediction via Transformer Embeddings and Bitscore-Weighted Features

The paper is under review process.

## Description
SUPERMAGOv2 is a machine learning-based approach designed for protein function prediction using embeddings generated by Transformer-based models, bitscore-weighted features, multilayer perceptrons, ResNet50, and a stacking classifier. SUPERMAGOv2+ is an ensemble method that combines predictions from SUPERMAGOv2 and DIAMOND. Both approaches predict protein function for Biological Process Ontology (BPO), Cellular Component Ontology (CCO), and Molecular Function Ontology (MFO).

## Instalation
To install and set up SUPERMAGOv2 and SUPERMAGOv2+, follow the steps below:

1. **Clone the repository:**
```bash
git clone https://github.com/gabrielbianchin/SUPERMAGOv2.git
cd SUPERMAGOv2
```
2. **Install the dependencies:**
```bash
pip install -r requirements.txt
```

## Dataset
The dataset for this work is available [here](https://zenodo.org/records/10982903).
The IC values used in evaluation is available [here](https://zenodo.org/records/13362841).

## Models
Our layer-base models for each ontology are available here.

## Predictions
The predictions of SUPERMAGO and SUPERMAGO+ are available here.

## Reproducibility
* Navigate to ```src``` folder and run ```setup.py```.
* Download the dataset and IC values, and place them into ```base``` folder.
* In the ```src``` folder, run the following command:
```python
python main.py --ont ontology
```
where **ontology** can be ```bp```, ```cc```, or ```mf``` for Biological Process, Cellular Component and Molecular Function, respectively.

```main.py``` executes the pipeline of SUPERMAGOv2 and SUPERMAGOv2+ as follows:
* ```extract.py``` extracts the embeddings given the model name (```esm``` or ```t5```) and ontology (```bp```, ```cc``` or ```mf```).
* ```diamond.py``` runs DIAMOND predictions and bitscore-weighted features for a specific ontology (```bp```, ```cc``` or ```mf```).
* ```mlp.py``` runs the MLP for a specific layer (```36```, ```35```, ```34```, ```24```, ```23```, or ```22```) and ontology (```bp```, ```cc``` or ```mf```).
* ```resnet50.py``` runs ResNet50 for a specific ontology (```bp```, ```cc``` or ```mf```).
* ```stacking.py``` runs the stacking model for a specific ontology (```bp```, ```cc``` or ```mf```) and generates the prediction of SUPERMAGOv2.
* ```ensemble.py``` generates the final prediction of SUPERMAGOv2+ for a specific ontology (```bp```, ```cc``` or ```mf```).

## Dataset Adaptation
If you need to run SUPERMAGOv2 and SUPERMAGOv2+ on your own dataset, you must create a dataset with the same structure as ours. This includes a CSV file for each ontology, with the first column containing the protein ID, the second column containing the protein sequence, and the remaining columns containing terms in one-hot encoding format. You should also calculate the IC values for evaluation and save it in a csv file.
