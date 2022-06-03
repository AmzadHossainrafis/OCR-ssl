from dataloader import DataLoader 
#import load model from keras 
from train import valid_ds
from keras.models import load_model
from utils import read_yaml
from model import build_model
import tensorflow as tf



config=read_yaml("config.yaml")
model=build_model()
#model=model.load_model(config["model_dir"])
# bit confuse what to do .. i need to compile the model but i already compiled it in build_model()
model.load_weights(config["model_dir"])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
model.evaluate(valid_ds)    