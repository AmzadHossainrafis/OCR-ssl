import numpy as np
import matplotlib.pyplot as plt
import os 
import yaml
from pathlib import Path
import tensorflow as tf
import cv2
from PIL import Image
from tensorflow.keras import layers



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


def data_preprocess(images, labels,characters):
    """
    Preprocess the images and labels.
    :param images: The images.
    :param labels: The labels.
    :return: The preprocessed images and labels.
    """

    char_to_num = layers.StringLookup(
    vocabulary=list(characters), mask_token=None
)
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
    img=img.T
    label=tf.strings.unicode_split(labels, input_encoding="UTF-8")
    char_to_num(label)
    return {"image": img, "label": label}







    # img = tf.io.read_file(img_path)
    # # 2. Decode and convert to grayscale
    # img = tf.io.decode_png(img, channels=1)
    # # 3. Convert to float32 in [0, 1] range
    # img = tf.image.convert_image_dtype(img, tf.float32)
    # # 4. Resize to the desired size
    # img = tf.image.resize(img, [img_height, img_width])
    # # 5. Transpose the image because we want the time
    # # dimension to correspond to the width of the image.
    # img = tf.transpose(img, perm=[1, 0, 2])
    # # 6. Map the characters in label to numbers
    # label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # # 7. Return a dict as our model is expecting two inputs
    # 
 