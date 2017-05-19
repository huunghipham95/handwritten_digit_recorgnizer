from __future__ import print_function
import keras
import worker
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from keras.models import model_from_json

class MnistModel(object):
    def __init__(self):
        json_file = open('models/simplemnist_600.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights("models/simplemnist_600.h5")
        self.loaded_model.compile(loss=keras.losses.categorical_crossentropy,
                             optimizer=keras.optimizers.Adadelta(),
                             metrics=['accuracy'])
    def predict(self, bitmap):
        bitmap = np.expand_dims(bitmap, axis=2)
        bitmap = np.expand_dims(bitmap, axis=0)
        result = self.loaded_model.predict(bitmap)
        predict_num = np.argmax(result)
        return {'result':result,'predict_num':predict_num}

