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


    def __data_preprocess(self,images, labels,characters):
        """
        Preprocess the images and labels.
        :param images: The images.
        :param labels: The labels.
        :return: The preprocessed images and labels.
        """

        char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)   
        img=cv2.imread(images,cv2.IMREAD_GRAYSCALE) # 1. Read image
        img=cv2.resize(img,(200,50))                # 2. Resize image
        img=np.array(img)                           # 3. Convert image to numpy array 
        img=img.astype(np.float32)                  # convrt to data type float32
        img=img/255                                 # 4. Normalize image
        img=np.transpose(img,(2,0,1))               # transpose image
        label=tf.strings.unicode_split(labels, input_encoding="UTF-8")
        char_to_num(label)
        return  img,label

    def __len__(self) -> int:
        return int(np.ceil(len(self.dataS) / float(self.batch_size)))

    def __getitem__(self, idx) -> tuple:
        batch_x = self.dataS[idx * self.batch_size:(idx + 1) * self.batch_size]  # get the batch of images
        batch_lable = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size] # get the batch of labels

        img=np.zeros((self.batch_size,200,50,1)) # create a numpy array of zeros to hold the images
        lable=[]
        #fro loop for img and lables 
        for i in range(len(batch_x)):
            imgs,lable=self.__data_preprocess(batch_x[i],batch_lable[i],self.characters)
            img[i]=imgs
            lable.append(lable)

        return img , lable


if __name__ == "__main__":
    # create a dataloader object
    dataloader = DataLoader(batch_size=32,data=x_train,labels=y_train,characters=characters)
    # create a generator object
    x,y=dataloader[0]
    print(x.shape)
    print(y[:10])