# Datasets

This directory contains all the datasets used for training, validation, and testing. The table below summarieses how the datasets are used. To download any of the datasets, use the corresponding link.

- `/annotations`, contains the annotations for each dataset. I cannot upload/share the annotations as they were obtained from Rayson Laroca, the author of this [paper](https://arxiv.org/abs/1909.01754), and is only available upon request directly from the author(s) of the paper, however, some datasets include the annotations when downloaded.

- `datasets_utils.py`, used to provide utility functions to operate on the datasets.
- `visualise_dataset.ipynb`, used to visualise, extract, and check the datasets and the image annotations/labels.
- `dataset_analysis.ipynb`, used to analyse the datasets and give an overview of key metrics.
- `darknet_dataset_preperations.ipynb`, used to make all datasets compatible with the darknet framework to be used for YOLOv4 training.

| Dataset      | # Samples | Resolution | Country     | Train % | Val % | Test % | Notes | Link                                                                      |
|--------------|-----------|------------|-------------|---------|-------|--------|------| ---------------------------------------------------------------------------|
| Caltech Cars | 124      | 896x592  | American |       |     |      | car, rear view | [Link](https://www.robots.ox.ac.uk/~vgg/data/cars_markus/cars_markus.tar) |
| English LP | 509      | Mixed  | EU  |       |     |      | car/truck, rear view | [Link](http://www.zemris.fer.hr/projects/LicensePlates/english/baza_slika.zip) |
| OpenALPR EU | 108     | Mixed  | EU  |       |     |      | car, rear/front view | [Link](https://github.com/openalpr/benchmarks/tree/master/endtoend/eu) |
| AOLP | 2049     | Mixed  | Taiwanese  |       |     |      | cars/motorbikes, r/f | [Link](https://github.com/HaoRecog/AOLP) |
| UFPR-ALPR | 4500     | 1920x1080  | Brazilian  |       |     |      | car/motorbike, rear | [Link](https://web.inf.ufpr.br/vri/databases/ufpr-alpr/license-agreement/) |


To-do:
- [ ] Add the new Indian dataset to the table.
- [ ] Analyse the new data.
- [ ] Label the new data.
- [ ] Further document `datasets_utils.py`

Waiting for additional dataset:
- SSIG-SegPlate, request document requires signature from the head of department, I have contacted Nastasha regarding this.