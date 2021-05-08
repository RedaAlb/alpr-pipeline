# Vehicle type classfication (VTC)

- For the VTC, a ResNet50 model was used to classify images as either an emergency vehicle or a truck. There is also a third class "other", which refers to all other types of vehicles, for example, cars and motorcycles.
- Data augmentations of rotation, width and height shifts, brightness, shear, zoom, and horizontal flip was used.
- Transfer learning was used where the weights are the ResNet50 model trained on the COCO dataset, all the ResNet50 layers are frozen, and average pooling was added to the end, followed by a fully connected layer of 32, followed by the softmax output layer.


# Notes

- `/datasets`, contains information on how to obtain the dataset used.
- `/saved_models`, where all the models will be saved, I have included a link for you to download the final trained model used.
- `evaluate_model.ipynb`, used to obtain the train, val, test accuracies and losses, and get the miss-classified samples by the model.
- `helper.py`, provides helper functions.
- `VTC_ResNet_training.py`, used to train the model.