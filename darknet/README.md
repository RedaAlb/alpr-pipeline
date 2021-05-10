# Darknet

Install [Darknet](https://github.com/AlexeyAB/darknet) in this directory by following the instructions in the Darknet repo.


# How to run the ALPR pipeline:

To get the same configurations and weights used, please refer to the `/saved_data` directory.

1. After installing Darknet, copy the following files to your build directory (`build/darknet/x64`):
	- `run_alpr.py`, used to run the alpr pipeline.
	- `alpr_pipeline.py`, implements the full ALPR pipeline, first detects the vehicles in the frame, crops the vehicles, then for each vehicle patch the LP is detected, each LP patch is cropped and then the LP characters are detected and assembled to form the full LP text.
	- `detector.py`, used to create YOLO detectors to be easily used.
	- `evaluator.py`, used to evaluate each stage of the ALPR pipeline, vehicle detection, LP detection, LP recognition.
2. Open `run_alpr.py`, change the variables/configurations to your needs.
3. Run `run_alpr.py`.


# Some notes about training a custom darknet model

More information regarding this can be found in the [Darknet repo](https://github.com/AlexeyAB/darknet). Here I have summarised the key aspects.

- Download the required pre-trained weights e.g. `yolov4.conv.137` or `yolov4-tiny.conv.29` from the Darknet repo.
- Make a copy of `yolov4-custom.cfg`, for yolo-tiny `yolov4-tiny-custom.cfg`.
- In the `.cfg` file, make these changes:
- `batch=64`.
- `subdivisions=16`.
- `max_batches` to (`classes*2000` but not less than number of training images, but not less than number of training images and not less than `6000`).
- `steps` to 80% and 90% of `max_batches`.
- `width` and `height` to any desired value (multiple of 32).
- For each `[yolo]` layer, change `classes=` to number of classes.
- For each `[convolutional]` before each `[yolo]` layer, change `filters=(classes + 5) * 3`

---

- Create `obj.names` in the `build\darknet\x64\data\` directory, with class names on each line.
- Create `obj.data` in the `build\darknet\x64\data\` directory, containing:

```
classes = 2
train  = data/train.txt
valid  = data/val.txt
names = data/obj.names
backup = backup/
```

---

- Regarding the dataset:
	- Each image should have a `.txt` with the same image name in the same directory with each line:
		- `<object-class> <x_center> <y_center> <width> <height>`, where
		- `<object-class>` - integer object number from 0 to (classes-1).
		- `<x_center> <y_center> <width> <height>` - float values relative to width and height of image, it can be equal from `(0.0 to 1.0]`.
			- for example: `<x> = <absolute_x> / <image_width>` or `<height> = <absolute_height> / <image_height>`.
			- Attention: `<x_center> <y_center>` - is the center of the rectangle (not top-left corner).
		- Example:
		```
		1 0.716797 0.395833 0.216406 0.147222
		0 0.687109 0.379167 0.255469 0.158333
		1 0.420312 0.395833 0.140625 0.166667
		```
	- Create `train.txt` and `val.txt` containing image paths relative to `darknet.exe`, place it in `build\darknet\x64\data`.

- To disable `flip` augmentation, add `flip=0` just under the `hue=.1` line in the `.cfg` file.

- To recalculate anchors for your dataset for `width` and `height` from cfg-file:
`darknet.exe detector calc_anchors data/obj.data -num_of_clusters 9 -width 416 -height 416`
then set the same 9 `anchors` in each of 3 `[yolo]`-layers in your cfg-file. But you should change indexes of anchors `masks=` for each [yolo]-layer, so for YOLOv4 the 1st-[yolo]-layer has anchors smaller than 30x30, 2nd smaller than 60x60, 3rd remaining, and vice versa for YOLOv3. Also you should change the `filters=(classes + 5)*<number of mask>` before each [yolo]-layer. If many of the calculated anchors do not fit under the appropriate layers - then just try using all the default anchors.

---

- Start training with:
	- For full YOLOv4, `darknet.exe detector train data/obj.data cfg/yolov4-obj.cfg yolov4.conv.137 -map`
	- For tiny, same steps but use `yolov4-tiny-custom.cfg`, and then `darknet.exe detector train data/obj.data cfg/yolov4-tiny-obj.cfg yolov4-tiny.conv.29 -map`


## During training

- `yolo-obj_last.weights` will be saved to the `build\darknet\x64\backup\` for each 100 iterations.
- `yolo-obj_xxxx.weights` will be saved to the `build\darknet\x64\backup\` for each 1000 iterations.
- When you should stop training:
	- Usually 2000 iterations for each class, but not less than the number of training images, and not less than 6000 in total.


## After training

- To detect, `darknet.exe detector test data/obj.data cfg/yolo-obj.cfg backup/yolo-obj_8000.weights`
- If the error `Out of memory` occurs during training, then increase `subdivisions=16` to 32 or 64.
- To calculate the `mAP` use:
	- `darknet.exe detector map data/obj.data cfg/yolov4-tiny-obj.cfg backup/0/yolov4-tiny-obj_2000.weights -thresh 0.75 -iou_thresh 0.5`
		- This will use the set whatever `valid` is in the `obj.data` file.
		- `-thresh` for the confidence threshold.
- WebCam: `darknet.exe detector demo data/coco.data cfg/yolov4.cfg yolov4.weights -c 0`


## How to improve detection

- Set the flag `random=1` in the cfg. It will train using different resolutions.
- Increase the network input resolution (`width` and `height`) (remember multiple of 32).
- You should aim for 2000 samples per class.
- Include negative samples, images that does not contain the object(s) you are detecting. Leave the `.txt` for those images blank.
	- Use as many negative samples as there are positive samples.
- Note that if you train on images where your object takes up most of the image (80% - 90%) then in your test set, if the object is only taking up 10%-20% of the image, then the performance will not be as good. So ensure that you have a mix in both datasets; images where the object takes up most of the image and where the object is a small part/patch in the image.
- After training, increase the input resolution to (`height=608` and `width=608`) or (`height=832` and `width=832`).
	- This will increase the precision and makes it possible to detect small objects.
	- Just change the `.cfg` file, no need to re-train.
- To get even greater results, if computationally feasible, train with higher resolution, e.g. 608x608 or 832x832, and if `Out of memory` error occurs, increase the `subdivisions` to 32 or 64.
