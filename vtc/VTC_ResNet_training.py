#%%
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

import numpy as np
import splitfolders

from helper import Helper

#%% Setup

DATA_DIR_PATH = "datasets/main"  # Where the dataset is located.
OUTPUT_DIR = "datasets/split_images"  # Where to save the trian, val, and test sets.

SEED = 22

TRAIN_R = 0.7  # Train ratio
VAL_R = 0.2
TEST_R = 0.1


EPOCHS = 200
MODEL_NAME = "vtc_final_model"

BATCH_SIZE = 32
IMG_SHAPE = (224, 224, 3)


# Data augmentation
D_AUG = True
ROTATION = 10
WIDTH_SHIFT = 0.2
HEIGHT_SHIFT = 0.2
BRIGHTNESS = (0.2, 1.4)
SHEAR = 0.2
ZOOM = 0.3
HORI_FLIP = True


# To split the dataset into train, val, and test sets.
splitfolders.ratio(DATA_DIR_PATH, OUTPUT_DIR, seed=SEED, ratio=(TRAIN_R, VAL_R, TEST_R))

train_data_dir = f"{OUTPUT_DIR}/train"
val_data_dir = f"{OUTPUT_DIR}/val"
test_data_dir = f"{OUTPUT_DIR}/test"


helper = Helper()

train_gen, val_gen, test_gen = helper.get_resnet_gens(train_data_dir,
                                                      val_data_dir,
                                                      test_data_dir,
                                                      target_size=(IMG_SHAPE[0], IMG_SHAPE[1]),
                                                      batch_size=BATCH_SIZE,
                                                      data_aug=D_AUG,
                                                      rotation=ROTATION,
                                                      width_shift=WIDTH_SHIFT,
                                                      height_shift=HEIGHT_SHIFT,
                                                      brightness=BRIGHTNESS,
                                                      shear=SHEAR,
                                                      zoom=ZOOM,
                                                      hori_flip=HORI_FLIP
                                                      )


#%% Creating the model

resnet_model = ResNet50(include_top=False, weights="imagenet", input_shape=IMG_SHAPE, pooling="avg")

# Adding final layers at the end of the model.
x = resnet_model.output
x = Dense(32, activation="relu")(x)

output = Dense(train_gen.num_classes, activation="softmax")(x)
model = Model(inputs=resnet_model.input, outputs=output)

# Freezing ResNet the layers.
for layer in resnet_model.layers:
    layer.trainable = False
    
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["acc"])

model.summary()

#%% Model training

history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=EPOCHS)

#%% Test set evaluation

print("\nTest evaluation:")
model.evaluate(test_gen)


#%% Saving the model

print("\nSaving model and history...")

model.save(f"saved_models/{MODEL_NAME}.h5")
    
history_array = np.array([history.history["loss"],
                          history.history["acc"],
                          history.history["val_loss"],
                          history.history["val_acc"]])

with open(f"saved_models/{MODEL_NAME}.npy", "wb") as file:
    np.save(file, history_array)
    
print("Model and history saved")
print("Done")
