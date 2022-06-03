
from pyexpat import model
from utils import * 
import numpy as np
import cv2
from model import *


data,labels,characters=read_img_lables()

x_train, x_valid, y_train, y_valid = split_data(np.array(data), np.array(labels))


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self,batch_size,data,labels,characters) :
        self.batch_size = batch_size
        self.dataS = data
        self.labels = labels
        self.characters = characters


    def __data_preprocess(self,images, labels,characters):
        """da
        Preprocess the images and labels.
        :param images: The images.
        :param labels: The labels.
        :return: The preprocessed images and labels.
        
            """
        
        
        char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
        img = tf.io.read_file(images)
        # 2. Decode and convert to grayscale
        img = tf.io.decode_png(img, channels=1)
        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size
        img = tf.image.resize(img, [img_height, img_width])
        # 5. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])
        # 6. Map the characters in label to numbers
        label = char_to_num(tf.strings.unicode_split(labels, input_encoding="UTF-8"))
        # 7. Return a dict as our model is expecting two inputs
        return img,label

    def __len__(self) -> int:
        return int(np.ceil(len(self.dataS) / float(self.batch_size)))

    def __getitem__(self, idx) -> tuple:
        batch_x = self.dataS[idx * self.batch_size:(idx + 1) * self.batch_size]  # get the batch of images
        batch_lable = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size] # get the batch of labels

        img=np.zeros((self.batch_size,200,50,1)) # create a numpy array of zeros to hold the images
        lables=[]
        sub=[]
        #fro loop for img and lables 
        for i in range(len(batch_x)):
            imgs,lable=self.__data_preprocess(batch_x[i],batch_lable[i],self.characters)
            img[i]=imgs
            lables.append(lable)

        return ([img,lables] , None)# get


if __name__ == '__main__':
# create a dataloader object
    train_ds = DataLoader(batch_size=2,data=x_train,labels=y_train,characters=characters)
    # create a generator object
    x,y=train_ds[0]


    val_ds= DataLoader(batch_size=2,data=x_valid,labels=y_valid,characters=characters)
    model=build_model()

    model.fit(train_ds, validation_data=val_ds, epochs=10)