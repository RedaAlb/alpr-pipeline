# Using Darket

This readme contains notes about using the Darknet framework and some general notes.

Install [Darknet](https://github.com/AlexeyAB/darknet) in this directory by following the instructions in the repo.


## Setup for training

- Download required pre-trained weights e.g. `yolov4.conv.137` or `yolov4-tiny.conv.29`.
- Make copy of `yolov4-custom.cfg` to `yolov4-obj.cfg`. Or any, e.g. for tiny `yolov4-tiny-custom.cfg`.
	- Just change custom to `obj` to easily distinguish them.
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

- Regarding the dataset:
	- Each image should have a `.txt` with the same image name in the same directory with each line:
		- `<object-class> <x_center> <y_center> <width> <height>`, where
		- `<object-class>` - integer object number from 0 to (classes-1).
		- `<x_center> <y_center> <width> <height>` - float values relative to width and height of image, it can be equal from `(0.0 to 1.0]`.
			- for example: `<x> = <absolute_x> / <image_width>` or `<height> = <absolute_height> / <image_height>`.
			- Attention: `<x_center> <y_center>` - are center of rectangle (are not top-left corner).
		- Example:
		```
		1 0.716797 0.395833 0.216406 0.147222
		0 0.687109 0.379167 0.255469 0.158333
		1 0.420312 0.395833 0.140625 0.166667
		```
	- Create `train.txt` containing image paths relative to `darknet.exe`, place it in `build\darknet\x64\data\`.

- To disable `flip` augmentation, add `flip=0` just under the `hue=.1` line in the `.cfg` file.

- To make detected BBs more accurate, you can add 3 parameters,
	- `ignore_thresh = .9`
	- `iou_normalizer=0.5`
	- `iou_loss=giou`
	- to each `[yolo]` layer. It will increase mAP@0.9, but decrease mAP@0.5.

- To recalculate anchors for your dataset for `width` and `height` from cfg-file:
`darknet.exe detector calc_anchors data/obj.data -num_of_clusters 9 -width 416 -height 416`
then set the same 9 `anchors` in each of 3 `[yolo]`-layers in your cfg-file. But you should change indexes of anchors `masks=` for each [yolo]-layer, so for YOLOv4 the 1st-[yolo]-layer has anchors smaller than 30x30, 2nd smaller than 60x60, 3rd remaining, and vice versa for YOLOv3. Also you should change the `filters=(classes + 5)*<number of mask>` before each [yolo]-layer. If many of the calculated anchors do not fit under the appropriate layers - then just try using all the default anchors.


- Start training with:
	- For full YOLOv4, `darknet.exe detector train data/obj.data cfg/yolov4-obj.cfg yolov4.conv.137 -map`
	- For tiny, same steps but use `yolov4-tiny-custom.cfg`, and then `darknet.exe detector train data/obj.data cfg/yolov4-tiny-obj.cfg yolov4-tiny.conv.29 -map`
	- Use the `-show_imgs` flag at end of training command to show BBs of objects.


## During training
- (file `yolo-obj_last.weights` will be saved to the `build\darknet\x64\backup\` for each 100 iterations)
- (file `yolo-obj_xxxx.weights` will be saved to the `build\darknet\x64\backup\` for each 1000 iterations)
- When you should stop training:
	- Usually 2000 iterations for each class, but not less than the # of training images, and not less than 6000 in total.

## After training
- To detect, `darknet.exe detector test data/obj.data yolo-obj.cfg yolo-obj_8000.weights`
- If the error `Out of memory` occurs during training, then increase `subdivisions=16` to 32 or 64.
- To calculate final `mAP` and to choose best model, use:
	- `darknet.exe detector map data/obj.data cfg/yolov4-tiny-obj.cfg backup/0/yolov4-tiny-obj_2000.weights -thresh 0.75 -iou_thresh 0.5`
		- This will use the set whatever val set is in the `obj.data` file.
		- `-thresh` for the confidence threshold.


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
- But to get even greater results, if computationally feasible, train with higher resolution, e.g. 608x608 or 832x832, and if `Out of memory` error occurs, increase the `subdivisions` to 32 or 64.


## Precision and Recall quick recap

- Calculating precision and recall.
	- Order all BBs based on their confidence scores, then for each BB, you classify it if its correct or not (1(TP), 0(FP) based on your confidence threshold.
	- Then for each row (BB), you keep track of the sum of correct(1, TP) and incorrect(0, FP) detections, and the precision and recall is calculated using:
		- `precision = TP / TP + FP`
		- `recall = TP / #_of_GT_BBs`
	- Example:
		- Consider an image with 12 dogs (relevant object) and 10 cats.
		- 8 dogs were detected
			- 5 actually dogs `TP`.
			- 3 were cats `FP`.
		- 7 dogs were missed `FN`.
		- 7 cats were correctly excluded `TN`.
		- `precision = 5/8 = TP / (TP + FP)`
		- `recall = 5/12 = TP / #_of_GT_BBs or (TP + FN)`	
- Then plot Precision x Recall graph.
	- You will get a "zig-zag" graph
- How to determine which model is better based on graph:
	- Using Average Precision (AP):
		- After we have the Precision x Recall graph, we use 11 point interpolation to get the AP.
		- We get the precision value for recall values of [0, 0.1, 0.2, ..., 1.0], sum them up and divide by 11 to get the average, and that is the AP.
		- Now for each model, we can use the AP for comparison.
	- Or calculate it using All point interpolation, where we get the area under the graph.
	- Using mean AP (mAP) (for multiple classes):
		- First calculate AP for each class.
		- Then take average of that. 
		- `mAP = (APs / # of classes)`.



## Tools
- To label images:
	- [labelImg](https://github.com/tzutalin/labelImg)


