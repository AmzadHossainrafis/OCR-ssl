
from utils import read_yaml,SelectCallbacks,read_img_lables,split_data
from dataloader import DataLoader
import numpy as np
from model import *

config=read_yaml("config.yaml")
model=build_model()
callbacks=SelectCallbacks()

data,labels,characters=read_img_lables()
x_train, x_valid, y_train, y_valid = split_data(np.array(data), np.array(labels))

train_ds = DataLoader(batch_size=config["batch_size"],data=x_train, labels= y_train,characters=characters)
valid_ds = DataLoader(batch_size=config["batch_size"],data=x_valid, labels= y_valid,characters=characters)

history=model.fit(train_ds,validation_data=valid_ds,epochs=config["epochs"],callbacks=callbacks)