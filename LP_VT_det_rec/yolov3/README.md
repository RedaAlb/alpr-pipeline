# YOLOv3

- This directory is for all the implementations of YOLOv3 tested.

## To run the different YOLOv3 TensorFlow implementations

Below is a list of all the implementations tested.

Navigate to one of the YOLO directories and follow instructions on the corresponding readme file.

1. `/yolov3/yolov3_experiencor/`, runs very slow at ~3 FPS. [Original repo](https://github.com/experiencor/keras-yolo3).

1. `/yolov3/yolov3_ml_space/`, runs on average at ~8 FPS. [Original repo](https://github.com/RahmadSadli/Deep-Learning).

1. `/yolov3/yolov3_zzh/`, runs on average at ~12 FPS. [Original repo](https://github.com/zzh8829/yolov3-tf2).


## To run LP detector and recogniser, Automatic number-plate recognition (ALPR)

For the ALPR, the method that is used as the starting point is this [proposed method](https://www.groundai.com/project/an-efficient-and-layout-independent-automatic-license-plate-recognition-system-based-on-the-yolo-detector/1). To run it, follow the steps below.

Requirements:
- Linux system.
- [Darknet](https://github.com/AlexeyAB/darknet) installed in the `/darknet` directory.

Once Darknet is installed and compiled successfully:

1. Navigate to the `/darknet` directory.
2. Download the required files:

```
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/vehicle-detection.cfg
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/vehicle-detection.data
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/vehicle-detection.weights
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/vehicle-detection.names
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/lp-detection-layout-classification.cfg
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/lp-detection-layout-classification.data
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/lp-detection-layout-classification.weights
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/lp-detection-layout-classification.names
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/lp-recognition.cfg
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/lp-recognition.data
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/lp-recognition.weights
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/lp-recognition.names
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/sample-image.jpg
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/README.txt
```

3. Detect the vehicles in the image:
```python
./darknet detector test vehicle-detection.data vehicle-detection.cfg vehicle-detection.weights -thresh .25 <<< sample-image.jpg
```

4. LP detection and layout classification

Provided that the vehicles detected are cropped, run the following command for each cropped patch.

```python
./darknet detector test lp-detection-layout-classification.data lp-detection-layout-classification.cfg lp-detection-layout-classification.weights -thresh .01 <<< motorcycle.jpg
./darknet detector test lp-detection-layout-classification.data lp-detection-layout-classification.cfg lp-detection-layout-classification.weights -thresh .01 <<< car.jpg
```

5. Finally, for each LP patch, recognise the characters by running the command below.

```python
./darknet detector test lp-recognition.data lp-recognition.cfg lp-recognition.weights -thresh .5 <<< lp-motorcycle.jpg
./darknet detector test lp-recognition.data lp-recognition.cfg lp-recognition.weights -thresh .5 <<< lp-car.jpg
```

<br>
<br>
<br>
<br>
<br>

*Note, this will be updated as the complete LP and VT detection and recongition method evolves towards the end goal of this part of the project, which is to detect/recognise VT and Indian LPs.