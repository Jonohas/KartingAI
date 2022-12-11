# =========================== IMPORTS ===========================
from os import walk, path, environ, getenv
from random import shuffle
import cv2
from skimage import data, color, io, filters, morphology,transform, exposure, feature, util
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tqdm import tqdm
from datetime import date, datetime
import shutil

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import TensorBoard
# =========================== IMPORTS ===========================

# Setup tensorboard
tensorboard = TensorBoard(log_dir="./logs")


# Use GPU
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

load_dotenv()



def rescale(training_images, test_images):
    train_pb = tqdm(total=len(training_images), desc='Rescaling training images')
    for i in range(0,len(training_images)):
        train_pb.set_description_str(f"{training_images[i].shape}")
        training_images[i] = transform.resize(training_images[i],(imgrows,imgcols,3), mode='constant')
        train_pb.update(1)

    test_pb = tqdm(total=len(test_images), desc='Rescaling test images')
    for i in range(0,len(test_images)):
        test_pb.set_description_str(f"{test_images[i].shape}")
        test_images[i] = transform.resize(test_images[i],(imgrows,imgcols,3), mode='constant')
        test_pb.update(1)

    return training_images, test_images

def read_and_rescale():
    training_images, y_train, test_images, y_test = read_images()
    # training_images, test_images = rescale(training_images, test_images)
    return training_images, y_train, test_images, y_test

def train_test_split(training_images, y_train, test_images, y_test):
    training_images = np.asarray(training_images)
    test_images = np.asarray(test_images)

    X_train = training_images.reshape((len(training_images),imgrows,imgcols,3))
    X_test = test_images.reshape((len(test_images),imgrows,imgcols,3))

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    return X_train, X_test, y_train, y_test

def categorical(X_train, X_test, y_train, y_test):
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # X_train /= 255
    # X_test /= 255

    # for i in y_train:
    #     print(i)

    y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test - y_test.min())
    return X_train, y_train

def train(X_train, y_train):
    print("Training model...")
    lr = float(environ.get('LEARNING_RATE'))
    adam = tf.keras.optimizers.Adam(learning_rate = lr)

    # Neural network parameters
    #-----------------------------------------------
    #-----------------------------------------------
    batch_size = getenv("BATCH_SIZE") # 
    epochs = getenv("EPOCHS") # 
    kernel_size=(3,3)
    #-----------------------------------------------
    #-----------------------------------------------
    num_classes = 2



    # model.summary()

    history = model.fit(X_train, y_train, batch_size=int(batch_size), epochs=int(epochs), verbose=1, callbacks = [tensorboard])
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")


    for root, dirs, files in walk("./output/models/current/"):
        for dir in dirs:
            shutil.move(f"{root}{dir}", f"./output/models/old/{dir}")
    model.save('./output/models/current/vgg19finishdetector256_' + dt_string + '')
    return model

def test(model, X_test, y_test):
    print("Testing model...")
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    print(y_pred)
    print('\n')
    print(classification_report(y_test, y_pred))

    cf = confusion_matrix(y_test, y_pred)

    print(cf)
    print(accuracy_score(y_test, y_pred) * 100) 

    for x, y_t, y_p in zip(X_test, y_test, y_pred):
        if y_t != y_p:
            cv2.imshow(f"{y_p}", x)
            cv2.waitKey(0)

if __name__ == "__main__":
    training_images, y_train, test_images, y_test = read_and_rescale()
    X_train, X_test, y_train, y_test = train_test_split(training_images, y_train, test_images, y_test)
    X_train, y_train = categorical(X_train, X_test, y_train, y_test)
    model = train(X_train, y_train) 
    test(model, X_test, y_test)