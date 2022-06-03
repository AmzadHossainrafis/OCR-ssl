import numpy as np
import os 
import yaml
from pathlib import Path
import tensorflow as tf
import cv2
from tensorflow.keras import layers
from tensorflow import  keras
import math

def read_yaml(path='config.yaml'):
    """
    Reads the yaml file and returns the data in a dictionary.
    args: 
    path(str) The path to the yaml file.
    return: The data in the yaml file.

    """
    with open(path, 'r') as stream:
        data_loaded = yaml.load(stream, Loader=yaml.FullLoader)
    return data_loaded



def read_img_lables():
    """
    Reads the images and labels from the given path.
    args: path(str) The path to the images and labels.

    return: The images and labels and unique labels .

    """
    data_dir = Path("./data/captcha_images_v2/")

# Get list of all the images
    images_dir = sorted(list(map(str, list(data_dir.glob("*.png")))))
    labels = []
    for i in sorted(os.listdir(data_dir)):
        m=i.split(".png")[0]
        labels.append(m)
    characters = set(char for label in labels for char in label)
    return images_dir, labels,characters

           

#ocs model data download
def split_data(images, labels, train_size=0.9, shuffle=True):
    """ Split data into training and test sets.
    args:

    images:(list) list of images dir 
    labels:(list) list of labels 
    train_size:(float) percentage of data to use for training
    shuffle: bool, whether to shuffle the data before splitting

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


def data_preprocess(images, labels,characters):
    """
    Preprocess the images and labels.
    :param images: The images.
    :param labels: The labels.
    :return: The preprocessed images and labels.
    """

    char_to_num = layers.StringLookup(
    vocabulary=list(characters), mask_token=None)
    # 1. Read image
    img=cv2.imread(images,cv2.IMREAD_GRAYSCALE)
    # 2. Resize image
    img=cv2.resize(img,(200,50))
    # 3. Convert image to numpy array
    img=np.array(img)
   # convrt to data type float32
    img=img.astype(np.float32)
    # 4. Normalize image
    img=img/255
    # transpose image
    img=np.transpose(img,(2,0,1))
    label=tf.strings.unicode_split(labels, input_encoding="UTF-8")
    char_to_num(label)
    return {"image": img, "label": label}



class SelectCallbacks(keras.callbacks.Callback):
    def __init__(self,config= read_yaml()):
        """
        Summary:
            callback class for validation prediction and create the necessary callbacks objects
        Arguments:
            val_dataset (object): MyDataset class object
            model (object): keras.Model object
            config (dict): configuration dictionary
        Return:
            class object
        """



        super(keras.callbacks.Callback, self).__init__()
        self.config = config
        self.callbacks = []

    def lr_scheduler(self, epoch):
        """
        Summary:
            learning rate decrease according to the model performance
        Arguments:
            epoch (int): current epoch
        Return:
            learning rate
        """


        drop = 0.5
        epoch_drop = self.config['epochs'] / 8.
        lr = self.config['learning_rate'] * math.pow(drop, math.floor((1 + epoch) / epoch_drop))
        return lr


    def get_callbacks(self):
        """
        Summary:
            creating callbacks based on configuration
        Arguments:
            val_dataset (object): MyDataset class object
            model (object): keras.Model class object
        Return:
            list of callbacks
        """


        
        if self.config['csv']:
            self.callbacks.append(keras.callbacks.CSVLogger(os.path.join(self.config['csv_log_dir'], self.config['csv_log_name']), separator = ",", append = False))
        if self.config['checkpoint']:
            self.callbacks.append(keras.callbacks.ModelCheckpoint(filepath=self.config['checkpoint_dir']+"next_frame_prediction.hdf5", save_best_only = True))
        if self.config['lr']:
            self.callbacks.append(keras.callbacks.LearningRateScheduler(schedule = self.lr_scheduler))
        
        
        
        return self.callbacks

