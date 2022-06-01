from gc import callbacks
from tkinter import S
from dataloader import Dataloader
import tensorflow as tf
import numpy as np
from utils import *
from model import *

config=read_yaml("config.yaml")
model=build_model()
callbacks=SelectCallbacks()

#train dataloader 
train_ds=Dataloader(config['batch_size'],train_data)

#valloader
val_ds=Dataloader(config['batch_size'],val_data)

history=model.fit(train_ds, validation_data=val_ds, epochs=config['epochs'], callbacks=callbacks)



