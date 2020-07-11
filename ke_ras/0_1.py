import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import sparse_categorical_crossentropy


train_labels = []
train_samples = []

'''
Background story:
A drug was tested on 2100 patients btw the ages of 13 and 100,
young = 13 - 64
old = 65 -100

50 young patient experienced side effect
50 old didn't
1000 old did
1000 young didn't


Here we generate the data manually using random int numbers and append appropriate labels

'''

# the outliers
for i in range(50):
    # generate 50 random int btw 13 and 64 then append to samples
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    # add labels 1 to them
    train_labels.append(1)

    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)

# the expected
for i in range(1000):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    # add labels 1 to them
    train_labels.append(0)

    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
# shuffle
train_samples, train_labels = shuffle(train_samples, train_labels)

# scale from range 13-100 to range 0-1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))
print(train_samples.shape)
print(scaled_train_samples.shape)


# MODEL
model = Sequential()
model.add(Dense(units=16, input_shape=(1,), activation="relu"))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(units=2, activation="softmax"))

model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics = ['accuracy'])
model.fit(x=scaled_train_samples, y=train_labels, batch_size=10, epochs=60, verbose=2, shuffle=True, validation_split=0.1)


model.save("model.h5")