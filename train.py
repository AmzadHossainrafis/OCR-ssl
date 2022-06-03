
from utils import read_yaml,SelectCallbacks,read_img_lables,split_data

from dataloader import DataLoader
import numpy as np
from model import *

config=read_yaml("config.yaml")
model=build_model()
callbacks=SelectCallbacks()

data,labels,characters=read_img_lables()

x_train, x_valid, y_train, y_valid = split_data(np.array(data), np.array(labels))
#train dataloader 
train_ds=Dataloader(config['batch_size'],x_train,y_train,characters)

#valloader
val_ds=Dataloader(config['batch_size'],x_valid,y_valid,characters)

history=model.fit(train_ds, validation_data=val_ds, epochs=config['epochs'], callbacks=callbacks)



