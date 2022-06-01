import numpy as np
import matplotlib.pyplot as plt
import os 
import yaml
from pathlib import Path
def read_yaml(path='config.yaml'):
    """
    Reads the yaml file and returns the data in a dictionary.
    :param path: The path to the yaml file.
    :return: The data in the yaml file.
    """
    with open(path, 'r') as stream:
        data_loaded = yaml.load(stream, Loader=yaml.FullLoader)
    return data_loaded



def read_img_lables(path):
    """
    Reads the images and labels from the given path.
    :param path: The path to the images and labels.
    :return: The images and labels.
    """
    data_dir = Path("./data/captcha_images_v2/")
    images_dir = [map(str, list(data_dir.glob("*.png")))]
    labels = []
    for i in os.listdir(path):
        m=i.split(".png")[0]
        labels.append(m)
    return images_dir, labels

           

#ocs model data download
def split_data(images, labels, train_size=0.9, shuffle=True):
    """ Split data into training and test sets.
    
    """

    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid


