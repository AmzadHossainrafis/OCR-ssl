

from numpy import save
from utils import *
from dataloader import DataLoader
from model import *
from Ctc import CTCLoss
import tensorflow 

config=read_yaml("config.yaml")
model=build_model2()
model.compile("adam",loss=CTCLoss)
callbacks=SelectCallbacks()
chk_point=tensorflow.keras.callbacks.ModelCheckpoint(r'C:\Users\Amzad\Desktop\keras_project\OCR-ssl\logs\weights\ocr.h5')

data,labels,characters=read_img_lables()
x_train, x_valid, y_train, y_valid = split_data(np.array(data), np.array(labels),train_size=config['split_ratio'],shuffle=True)

train_ds = DataLoader(batch_size=config["batch_size"],data=x_train, labels= y_train,characters=characters,CTC=False)
valid_ds = DataLoader(batch_size=config["batch_size"],data=x_valid, labels= y_valid,characters=characters,CTC=False)

history=model.fit(train_ds,validation_data=valid_ds,epochs=config["epochs"],callbacks=[chk_point],shuffle=config['shuffle'],verbose=2)