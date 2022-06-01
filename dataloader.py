from cProfile import label
from numpy import character
import tensorflow as tf 
import os
from utils import * 

data,labels,characters=read_img_lables("./data/captcha_images_v2/")

x_train, x_valid, y_train, y_valid = split_data(np.array(data), np.array(labels))


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self,batch_size,data,labels,characters) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.dataS = data
        self.labels = labels
        self.characters = characters


    def __len__(self) -> int:
        return int(np.ceil(len(self.dataS) / float(self.batch_size)))

    def __getitem__(self, idx) -> tuple:


        return img , lable
