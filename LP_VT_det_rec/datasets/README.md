# Datasets

The table below summarieses all the datasets used. To obtain any of the datasets, use the corresponding link.

- `/annotations`, contains the annotations for each dataset. I am not permitted to upload/share the annotations as they were obtained from Rayson Laroca, one of the authors of this [paper]
(https://arxiv.org/abs/1909.01754), and is only available upon request directly from the author(s) of the paper, however, some datasets include the annotations when downloaded.
- `/class_names`, contains the class names used.
- `visualise_dataset.ipynb`, used to visualise and check the datasets with the image annotations/labels.
- `datasets_utils.py`, provides utility functions to operate on the datasets.
- `darknet_dataset_preperations.ipynb`, used to make all datasets compatible with the darknet framework to be used with YOLOv4.
- `data_generator.py`, used to generate new samples using various methods.
- `data_generation.ipynb`, applies what `data_generator.py` provides.
- `Automold.py` and `Helpers.py` are from this [repository](https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library). They are used in the data generation process.


## Main datasets

| Dataset      | # Samples | Resolution | Country   | Link                                                                           |
|--------------|-----------|------------|-----------|--------------------------------------------------------------------------------|
| Caltech Cars | 124       | 896x592    | American  | [Link](https://www.robots.ox.ac.uk/~vgg/data/cars_markus/cars_markus.tar)      |
| English LP   | 509       | Mixed      | EU        | [Link](http://www.zemris.fer.hr/projects/LicensePlates/english/baza_slika.zip) |
| OpenALPR EU  | 108       | Mixed      | EU        | [Link](https://github.com/openalpr/benchmarks/tree/master/endtoend/eu)         |
| AOLP         | 2049      | Mixed      | Taiwanese | [Link](https://github.com/HaoRecog/AOLP)                                       |
| UFPR-ALPR    | 4500      | 1920x1080  | Brazilian | [Link](https://web.inf.ufpr.br/vri/databases/ufpr-alpr/license-agreement/)     |
| **Total**    | **7290**  |


### Average (w/h) ratio for LP patches

| Dataset      | Average ratio (w/h) |
| ------------ | ------------------- |
| Caltech Cars | 1.982               |
| English LP   | 4.031               |
| OpenALPR EU  | 3.667               |
| AOLP         | 2.017               |
| UFPR-ALPR    | 2.597               |
| **Average**  | **2.859**           |



## To label a dataset (specifically for vehicle/LP detection and recognition):

- Use this [repo](https://github.com/RedaAlb/labelImg).
- Two videos were made demonstrating how the labelling is done:
    - [Part 1 video](https://youtu.be/5tF9a6q4pDQ)
    - [Part 2 video](https://youtu.be/YAxl1udnBqI)
    - There are two parts since the maximum upload video duration is 15 minutes.