from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

np.random.seed(3)

# number of wine classes
classifications = 3

# load dataset
dataset = np.loadtxt('wine.data', delimiter=",")

# split dataset into sets for testing and training
X = dataset[:,1:14]
Y = dataset[:,0:1]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.66, random_state=5)

# convert output values to one-hot
y_train = keras.utils.to_categorical(y_train-1, classifications)
y_test = keras.utils.to_categorical(y_test-1, classifications)


# creating model
model = Sequential()
model.add(Dense(13, input_dim=13, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(classifications, activation='softmax'))

# compile and fit model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=15, epochs=2500, validation_data=(x_test, y_test))
