from tensorflow import keras
import numpy as np
from sklearn.utils import shuffle
from random import randint
model = keras.models.load_model("model.h5")
test_samples = []
test_labels = []

# create test data
for i in range(10):
    # generate 50 random int btw 13 and 64 then append to samples
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    # add labels 1 to them
    test_labels.append(0)
for i in range(10):
    # generate 50 random int btw 13 and 64 then append to samples
    random_older = randint(65, 100)
    test_samples.append(random_older)
    # add labels 1 to them
    test_labels.append(1)

test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_samples, test_labels = shuffle(test_samples, test_labels)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))

predict = model.predict(x=scaled_test_samples, batch_size=10, verbose=1)

a = 0
for i in predict:
    print("AGE:{} - SIDE EFFECT:{} ".format( test_samples[a], np.argmax(i)))
    a = a+1