import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess


class Helper:
    """ Provides helper functions for the ResNet VTC model.
    """

    def __init__(self):
        pass


    def plot_loss_acc(self, MODEL_NAME):
        """ Plot model loss and accuracy graphs.

        Args:
            MODEL_NAME (str): The name of the model to show plots for.
        """

        with open(f"saved_models/{MODEL_NAME}.npy", "rb") as file:
            history = np.load(file)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history[0])
        plt.plot(history[2])
        plt.legend(["training", "validation"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss plot")

        plt.subplot(1, 2, 2)
        plt.plot(history[1])
        plt.plot(history[3])
        plt.legend(["training", "validation"])
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy plot")

    
    def get_resnet_gens(self,
                        train_data_dir,
                        val_data_dir,
                        test_data_dir,
                        target_size=(224, 224),
                        batch_size=32,
                        data_aug=False,
                        rotation=10,
                        width_shift=0.2,
                        height_shift=0.2,
                        brightness=(0.2, 1.4),
                        shear=0.2,
                        zoom=0.3,
                        hori_flip=True):

        """ Get the train, val, and test tf image data generators.

        Args:
            train_data_dir ([type]): Where the training samples are located.
            val_data_dir ([type]): Where the validation samples are located.
            test_data_dir ([type]): Where the testing samples are located.
            target_size (tuple, optional): Model image size. Defaults to (224, 224).
            batch_size (int, optional): The batch size. Defaults to 32.
            data_aug (bool, optional): Whether to use data augmenation (DA) or not. Defaults to False.
            rotation (int, optional): Rotation angle for the DA. Defaults to 10.
            width_shift (float, optional): Width shift for the DA. Defaults to 0.2.
            height_shift (float, optional): Height shift for the DA. Defaults to 0.2.
            brightness (tuple, optional): Brightness lower and upper limits for the DA. Defaults to (0.2, 1.4).
            shear (float, optional): Shear for the DA. Defaults to 0.2.
            zoom (float, optional): Zoom for the DA. Defaults to 0.3.
            hori_flip (bool, optional): Whether to apply horizontal flip in the DA. Defaults to True.

        Returns:
            tuple: All three image data generators for the trian, val, and test sets.
        """

        if data_aug:
            train_datagen = ImageDataGenerator(
                preprocessing_function=resnet_preprocess,
                rotation_range=rotation,
                width_shift_range=width_shift,
                height_shift_range=height_shift,
                brightness_range=brightness,
                shear_range=shear,
                zoom_range=zoom,
                horizontal_flip=hori_flip
                )

        else:
            train_datagen = ImageDataGenerator(
                preprocessing_function=resnet_preprocess,
            )

        val_test_datagen = ImageDataGenerator(
            preprocessing_function=resnet_preprocess,
        )

        train_gen = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode="categorical")

        val_gen = val_test_datagen.flow_from_directory(
            val_data_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode="categorical")

        test_gen = val_test_datagen.flow_from_directory(
            test_data_dir,
            target_size=target_size,
            batch_size=1,
            class_mode="categorical")
        
        return train_gen, val_gen, test_gen


    def evaluate_model(self, data_dir, model_name):
        """ Get the training, validation and testing accuracy and loss metrics with the plots.

        Args:
            data_dir (str): The root of the split dataset directory.
            model_name (str): The model name to evaluate.
        """

        train_data_dir = f"{data_dir}/train"
        val_data_dir = f"{data_dir}/val"
        test_data_dir = f"{data_dir}/test"

        train_gen, val_gen, test_gen = self.get_resnet_gens(train_data_dir, val_data_dir, test_data_dir, batch_size=1)

        self.plot_loss_acc(model_name)


        model = tf.keras.models.load_model(f"saved_models/{model_name}.h5")

        print("\nTraining evaluation:")
        model.evaluate(train_gen)

        print("\nValidation evaluation:")
        model.evaluate(val_gen)

        print("\nTesting evaluation:")
        model.evaluate(test_gen)


    def get_wrong_samples(self, all_data_dir, model_name):
        """ Get the miss-classified images.

        Args:
            all_data_dir (str): The root dataset directory.
            model_name (str): The model name.
        """
        
        classes = sorted(os.listdir(all_data_dir))
        print("Classes:", classes)

        model = tf.keras.models.load_model(f"saved_models/{model_name}.h5")
        model_w, model_h = model.input.shape[1], model.input.shape[2]

        miss_class_imgs = []  # Will hold all the miss-classified images.

        for i, class_name in enumerate(classes):
            print(f"\nFor the ({class_name}) class:")

            img_names = os.listdir(f"{all_data_dir}/{class_name}")

            for j, img_name in enumerate(img_names):
                full_img_path = f"{all_data_dir}/{class_name}/{img_name}"

                img = plt.imread(full_img_path)
                img = cv2.resize(img, (model_w, model_h))
                img = np.array(img).reshape((1, model_w, model_h, 3))
                proc_img = resnet_preprocess(img)

                pred = list(model.predict(proc_img)[0])

                class_index = pred.index(max(pred))

                if class_index != i:
                    print(i, j, "Miss classification:", class_name, img_name)
                    miss_class_imgs.append(img)

                if j % 100 == 0 and j != 0:
                    print(j, "samples done")

        # Plot the miss-classified samples.
        for miss_img in miss_class_imgs:
            plt.figure()
            plt.imshow(miss_img[0])
