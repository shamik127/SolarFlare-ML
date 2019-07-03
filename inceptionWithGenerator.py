from __future__ import print_function

import os.path
import glob

import keras
from keras import applications, metrics, layers, models, regularizers, optimizers
from keras.applications import ResNet50, Xception
from keras.applications.inception_v3 import InceptionV3
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import layer_utils, to_categorical
from keras.datasets import mnist


import numpy as np
import tensorflow as tf
import keras
import os
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.utils import layer_utils, to_categorical
from keras.utils.data_utils import get_file
from keras.initializers import glorot_uniform
import scipy.misc

import psutil
from sklearn.metrics import confusion_matrix

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


TRAIN = np.load("temp_train.npy")
VALIDATION = np.load("temp_validation.npy")
mean = np.load('temp_mean.dat')
std = np.load('temp_std.dat')



#train_mean_crossval = np.load("train_mean_crossval.npy")
#validation_mean_crossval = np.load("validation_mean_crossval.npy")
#train_std_crossval = np.load("train_std_crossval.npy")
#validation_std_crossval = np.load("validation_std_crossval.npy")

IMAGE_SIZE = 256

CHANNELS = 1

n_of_train_samples = len(TRAIN.item().get('0'))
n_of_val_samples = len(VALIDATION.item().get('0'))

global GLOB_TSS
global LOSSES
global STATS


def get_y_val(file):
    return float(os.path.split(file)[-1][-5])
    
def get_test_data(j):
    x_data = []
    y_data = []
    for f in VALIDATION.item().get(str(j)):
        this_x = np.load(f)
        this_x = this_x.reshape(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        this_x = ((this_x - mean) / std)
        x_data.append(this_x)
        y_data.append(get_y_val(f))
    x_data = np.array(x_data, dtype = np.float32)
    y_data = np.array(y_data, dtype = np.float32)
    return (x_data, y_data)

def calc_TSS(confmat, nClasses):
    stats = {"recall": np.array([]), "Accuracy": np.array([]), "TSS": np.array([])}
    for i in xrange(0, nClasses):
        tp = confmat[i,i]
        tn = 0.0
        for j in range (i+1,nClasses):
            tn += confmat[j,j]
        for j in range (0,i):
            tn += confmat[j,j]
        fp = confmat[:,i].sum() - tp
        fn = confmat[i,:].sum() - tp
        myscore = float(( tp + tn )) / ( tp + tn + fp + fn)
        stats["Accuracy"] = np.append(stats["Accuracy"], myscore)
        myscore = float( tp ) / (tp + fn )
        stats["recall"] = np.append(stats["recall"], myscore)
        myscore = float( tp ) / ( tp + fn ) - float( fp ) / ( fp + tn )
        stats["TSS"] = np.append(stats["TSS"], myscore)
    STATS.append(stats)
    GLOB_TSS.append(stats['TSS'][0])

cache = {}
def generator(j):
    data_x = []
    data_y = []
    i = 0
    while True:
        FILE = TRAIN.item().get(str(j))[i]
        if FILE in cache:
            data_x_sample = cache[FILE][0]
            data_y_sample = cache[FILE][1]
        else:
            shape = (256, 256, CHANNELS)
            data_x_sample = np.load(FILE)
            data_x_sample = ((data_x_sample - mean) / std)
            data_y_sample = get_y_val(FILE)

            if psutil.virtual_memory().percent < 85:
                cache[FILE] = [data_x_sample, data_y_sample]

        data_x.append(data_x_sample)
        data_y.append(data_y_sample)
        i += 1
        if i == len(TRAIN.item().get(str(j))):
            i = 0
            np.random.shuffle(TRAIN.item().get(str(j)))

        if BATCH_SIZE == len(data_x):
            ret_x = np.reshape(data_x, (len(data_x), IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
            ret_y = np.reshape(data_y, (len(data_y), CHANNELS))
            yield (ret_x, ret_y)
            data_x = []
            data_y = []

def validation_generator(j):
    data_x = []
    data_y = []
    i = 0
    while True: 

        FILE = VALIDATION.item().get(str(j))[i]

        if FILE in cache:
            data_x_sample = cache[FILE][0]
            data_y_sample = cache[FILE][1]
        else:
            shape = (256, 256, CHANNELS)
            data_x_sample = np.load(FILE)
            data_x_sample = ((data_x_sample - mean) / std)
            data_y_sample = get_y_val(FILE)
            
            if psutil.virtual_memory().percent < 85:
                cache[FILE] = [data_x_sample, data_y_sample]

        data_x.append(data_x_sample)
        data_y.append(data_y_sample)
        i += 1
        if i == len(VALIDATION.item().get(str(j))):
            i = 0
            np.random.shuffle(VALIDATION.item().get(str(j)))

        if BATCH_SIZE == len(data_x):
            ret_x = np.reshape(data_x, (len(data_x), IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
            ret_y = np.reshape(data_y, (len(data_y), CHANNELS))
            yield (ret_x, ret_y)
            data_x = []
            data_y = []    


class simplecallback(keras.callbacks.Callback):

    def __init__(self, j):
        self.logpath = os.getcwd()
        self.j = j

    def on_train_begin(self, logs={}):
        self.losses = [[], []]
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses[0].append(logs.get("loss"))
        self.losses[1].append(logs.get("val_loss"))
        
    def on_train_end(self, logs={}):
        """
        num = np.array([], dtype = np.int8)
        search_path = 'cnn_without_padding/%s/losses_*.npy' %(name)
        files = glob.glob(search_path)
        if files == []:
            N = 1
        else:
            for f in files:
                n = int(os.path.split(f)[-1][-5])
                num = np.append(num, n)
            N = np.max(num) + 1
        write_path = 'cnn_without_padding/%s/losses_%s.npy' %(name, N) 
        np.save(write_path, self.losses)
        """
        LOSSES.append(self.losses)
        print(self.j)
        (x_test, y_test) = get_test_data(self.j)
        y_pred = model.predict(x_test)
        y_pred = y_pred.squeeze()
        y_pred[y_pred < 0.5] = 0
        y_pred[y_pred >= 0.5] = 1
        print(y_pred)

        y_test = y_test.squeeze()
        print(y_test)
        confmat = confusion_matrix(y_test,y_pred)
        print(confmat)
        calc_TSS(confmat,2)


FILTERS = {
    'F1': (3,3),
    'F2': (5,5),
    'F3': (7,7)
}

dropout_val = 0.4

num_layer = 2

def InceptionModule(A_prev):
    X1 = Conv2D(8, FILTERS['F1'], strides = (1,1), padding = 'same')(A_prev)
    #X1 = Activation('relu')(X1)
    X1 = LeakyReLU(alpha=0.1)(X1)

    X2 = Conv2D(8, FILTERS['F2'], strides = (1,1), padding = 'same')(A_prev)
    #X2 = Activation('relu')(X2)
    X2 = LeakyReLU(alpha=0.1)(X2)

    X3 = Conv2D(8, FILTERS['F3'], strides = (1,1), padding = 'same')(A_prev)
    #X3 = Activation('relu')(X3)
    X3 = LeakyReLU(alpha=0.1)(X3)

    X4 = MaxPooling2D((5,5), strides = (1,1), padding='same')(A_prev)
    X4 = Conv2D(8, (1,1), strides = (1,1))(X4)
    #X4 = Activation('relu')(X4)
    X4 = LeakyReLU(alpha=0.1)(X4)

    X = concatenate([X1, X2, X3, X4], axis = 3)

    return X


def InceptionModel(input_shape, classes):
    X_input = Input(input_shape)

    X = Conv2D(8, (7,7), strides = (2,2))(X_input)
    #X = BatchNormalization(axis = 3)(X)
    X = LeakyReLU(alpha=0.1)(X)
    #X = Activation('relu')(X)
    X = MaxPooling2D(pool_size = (3,3), strides = (2,2))(X)
    X = Conv2D(16, (3,3), strides = (1,1))(X)
    #X = BatchNormalization(axis = 3)(X)
    X = LeakyReLU(alpha=0.1)(X)
    #X = Activation('relu')(X)
    X = MaxPooling2D(pool_size = (3,3), strides = (2,2))(X)
    for i in range(num_layer):
        X = InceptionModule(X)
        if (i == num_layer - 1):
            X = GlobalAveragePooling2D()(X)
        else:
            X = MaxPooling2D(pool_size = (3,3), strides = (2,2))(X)
    #X = InceptionModule(X)
    #X = MaxPooling2D(pool_size = (3,3), strides = (2,2))(X)
    #X = InceptionModule(X)
    #X = GlobalAveragePooling2D()(X)
    X = Dropout(dropout_val)(X)
    X = Dense(classes, activation = 'sigmoid')(X)

    model = Model(inputs = X_input, outputs = X)
    return model

BATCH_SIZE = 64

class_weight = {0: 1.,
                1: 4.5}
cross_val = 5

GLOB_TSS = []
LOSSES = []
STATS = []

learning = 0.000005
EPOCHS = 100

layer_number = 2

for i in range(cross_val):
    model = InceptionModel((IMAGE_SIZE, IMAGE_SIZE, CHANNELS), 1)

    print(model.summary())
    
    model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.Adam(lr = learning),
    metrics=['acc'])
    
    model.fit_generator(
    generator(i),
    steps_per_epoch=n_of_train_samples//BATCH_SIZE,
    validation_data=validation_generator(i),
    validation_steps=n_of_val_samples//BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=[simplecallback(i)])

print(GLOB_TSS)

num = np.array([], dtype = np.int8)
search_path = 'inception_with_generator/cross_vals_*.npy'
files = glob.glob(search_path)
if files == []:
    N = 1
else:
    for f in files:
        n = int(os.path.split(f)[-1][11:-4])
        num = np.append(num, n)
    N = np.max(num) + 1

parameters = {'lr': learning, 'batch size': BATCH_SIZE, 'number of training files': n_of_train_samples, 'number of validation files': n_of_val_samples, 'epochs': EPOCHS, 'class_weight': class_weight, 'cross validation': cross_val, 'inception_filters': FILTERS, 'dropout': dropout_val, 'batch_norm': 'no', 'inception_layers': num_layer, 'activation': 'leakyrelu'}


write_path = 'inception_with_generator/params_%s.npy' %(N) 
np.save(write_path, parameters)

write_path = 'inception_with_generator/cross_vals_%s.npy' %(N) 
np.save(write_path, GLOB_TSS)

CROSS_VAL = np.mean(GLOB_TSS)
write_path = 'inception_with_generator/cross_val_%s.npy' %(N) 
np.save(write_path, CROSS_VAL)

model_write_path = 'inception_with_generator/model_%s.h5' %(N)
model.save(model_write_path)

write_path = 'inception_with_generator/losses_%s.npy' %(N) 
np.save(write_path, LOSSES)

write_path = 'inception_with_generator/stats_%s.npy' %(N) 
np.save(write_path, STATS)

load_path = 'inception_with_generator/cross_val_%s.npy' %(N)
tss = np.load(load_path)
print('cross validation TSS : ', tss)


dev = np.std(GLOB_TSS)
print(dev)