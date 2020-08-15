from tensorflow import keras
import numpy as np
from sklearn.utils import shuffle
from random import randint
import json
model = keras.models.load_model("model.h5")
print(model.to_json())


model.summary()
print(model.weights)

model2 = model
model2.su