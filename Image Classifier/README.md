## Building a Image Classifier

### Project Overview

In this project, I built a Python application that can train an image classifier on a dataset, then use the trained model to predict new images. The project gave me hands-on experience with building and training deep learning models using the PyTorch library.

I used a pre-trained convolutional neural network to classify images, then fine-tuned the model to improve its accuracy. I also learned how to use transfer learning to apply an existing model to a new task, and how to save and load trained models.

The project consisted of the following steps:

1. Loading and preprocessing the image dataset
2. Training the image classifier on my dataset
3. Using the trained classifier to predict image content

### Project Files

The project included the following files:

1. `README.md`: Provides an overview of the project and instructions for running the application.
2. `train.py`: A Python script for training the image classifier.
3. `predict.py`: A Python script for using the trained classifier to predict new images.
4. `workspace-utils.py`: A Python script containing utility functions used by the training and prediction scripts.
5. `cat_to_name.json`: A JSON file mapping category labels to category names.


### Arguments

The `train.py` and `predict.py` scripts accept the following arguments:

1. `--data_dir`: The path to the dataset directory.
2. `--save_dir`: The path to the directory where the trained model will be saved.
3. `--arch`: The architecture of the pre-trained model to use (e.g., "vgg16").
4. `--learning_rate`: The learning rate for training the model.
5. `--hidden_units`: The number of hidden units in the classifier.
6. `--epochs`: The number of epochs to train the model.
7. `--gpu`: Use GPU for training (default is CPU).
8. `--top_k`: The number of top K most likely classes for prediction.
9. `--category_names`: The path to a JSON file mapping category labels to category names.
10. `--img_path`: Path of the image.
11. `--checkpoint`: Select the saved model checkpoint.

### Credits

This project is part of the 'AI Programming with Python' Nanodegree from Udacity. The project starter code was provided by Udacity.