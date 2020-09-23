import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


from py_src.yolov4.tf import YOLOv4
from tensorflow.keras import optimizers
import cv2 as cv
import numpy as np
import time



yolo = YOLOv4()
yolo.classes = "test/coco.names"
yolo.make_model()
yolo.load_weights("weights/yolov4.weights", weights_type="yolo")

yolo.inference(media_path="test/kite.jpg")
# yolo.inference(media_path="test/road.mp4", is_image=False)
# yolo.inference(media_path=0, is_image=False)




# YOLOv4-tiny
# yolo = YOLOv4(tiny=True)
# yolo.classes = "test/coco.names"

# yolo.make_model()
# yolo.load_weights("weights/yolov4-tiny.weights", weights_type="yolo")

# yolo.inference(media_path="test/kite.jpg")
# yolo.inference(media_path="test/road.mp4", is_image=False)
# yolo.inference(media_path=0, is_image=False)
