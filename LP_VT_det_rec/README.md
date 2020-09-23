
# General notes
- [YOLO](https://pjreddie.com/darknet/yolo/) is a central part for this component of the project, and will be the main detection and recognition method used.
- Since the original YOLO algorithm is implemented using the [DarkNet](https://pjreddie.com/darknet/) framework. There are multiple implementations adapting the same exact YOLO algorithm to the more popular framework TensorFlow.
- First, to ensure the best and appropriate YOLO TensorFlow implementation is used, mulitple YOLO TensorFlow 2.0+ implementations are tested.
- Both YOLOv3 and YOLOv4 are tested.
- For YOLOv4, refer to the `/yolov4` directory.
- For YOLOv3, refer to the `/yolov3` directory.


# System specification used
- CPU: i7-7700K<br>
- GPU: Nvidia GTX 1060 6GB