# Datasets

This directory contains all the datasets used for training, validation, and testing. The table below summarieses how the datasets are used. To download any of the datasets, use the corresponding link.

- `/annotations`, contains the annotations for each dataset. I cannot upload/share the annotations as they were obtained from Rayson Laroca, the author of this [paper](https://arxiv.org/abs/1909.01754), and is only available upon request directly from the author(s) of the paper.

- `datasets_util.py`, used to provide utility functions to operate on the datasets.
- `visualise_dataset.ipynb`, used to visualise, extract, and check the datasets and the image annotations/labels.
- `dataset_analysis.ipynb`, will be used to analyse the datasets and give an overview of key metrics.


| Dataset      | # Samples | Resolution | Country     | Train % | Val % | Test % | Notes | Link                                                                      |
|--------------|-----------|------------|-------------|---------|-------|--------|------| ---------------------------------------------------------------------------|
| Caltech Cars | 126      | 896x592  | American |       |     |      | car, rear view | [Link](https://www.robots.ox.ac.uk/~vgg/data/cars_markus/cars_markus.tar) |
| English LP | 509      | Mixed  | EU  |       |     |      | car/truck, rear view | [Link](http://www.zemris.fer.hr/projects/LicensePlates/english/baza_slika.zip) |
| OpenALPR EU | 108     | Mixed  | EU  |       |     |      | car, rear/front view | [Link](https://github.com/openalpr/benchmarks/tree/master/endtoend/eu) |
| AOLP | 2049     | Mixed  | Taiwanese  |       |     |      | cars/motorbikes, r/f | [Link](https://github.com/HaoRecog/AOLP) |
| UFPR-ALPR | 4500     | 1920x1080  | Brazilian  |       |     |      | cars/motorbikes, rear | [Link](https://web.inf.ufpr.br/vri/databases/ufpr-alpr/license-agreement/) |



Waiting for additional dataset:
- SSIG-SegPlate, request document requires signature from the head of department, I have already contacted Nastasha regarding this.
- Real scenario dataset from Jude.