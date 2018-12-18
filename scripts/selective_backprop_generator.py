
# SB imports
from importance_sampling.training import SB

# Model imports
import numpy as np
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

num_classes = 10

def base_model(num_classes, x_train):

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr = 0.1, decay=1e-6, momentum=0.9, nesterov=True)

    return model

(x_train, y_train), (x_val, y_val) = cifar10.load_data()
model = base_model(num_classes, x_train)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Normalize to [0, 1]
x_train = x_train.astype(np.float32) / x_train.max()
x_val = x_val.astype(np.float32) / x_val.max()

y_val_categorical = keras.utils.to_categorical(y_val, num_classes)
y_train_categorical = keras.utils.to_categorical(y_train, num_classes)

train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(x_train[:40000], y_train_categorical[:40000], batch_size=128)
validation_generator = validation_datagen.flow(x_train[40000:], y_train_categorical[40000:], batch_size=128)

SB(model).fit_generator(
        train_generator,
        validation_data=validation_generator,
        validation_steps=len(x_train[40000:]) / 128,
        steps_per_epoch=len(x_train[:40000]) / 128,
        epochs=15,
        verbose=2,
        batch_size=128
)


model.evaluate(x_val, y_val_categorical)
