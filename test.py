from pickletools import optimize
from train import valid_ds
from keras.models import load_model
from utils import read_yaml
from Ctc import CTCLoss
import tensorflow as tf




config=read_yaml("config.yaml")
model=load_model(config['model_dir'])
model.compile('adam',loss=CTCLoss)
model.evaluate(valid_ds)    