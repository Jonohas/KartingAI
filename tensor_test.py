from os import walk, path
from random import shuffle
import cv2
from skimage import data, color, io, filters, morphology,transform, exposure, feature, util
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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

tensorboard = TensorBoard(log_dir="./logs")
from tqdm import tqdm


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

imgrows, imgcols = 144, 256

def read_images():
    training_images = []
    y_train = []
    test_images = []
    y_test = []

    shuffledvalidation = []
    shuffledtrain = []
    NF = []
    F = []
    for root, dirs, files in walk("./Labeled/"):
        for f in files:
            if 'notfinish' in f:
                NF.append(path.join(root,f))
            elif 'finish' in f:
                F.append(path.join(root,f))

    length = round(min(len(NF), len(F))*0.9)
    training_images += NF
    training_images += F
    shuffle(training_images)

    shuffledtrain = training_images[:length]
    shuffledvalidation = training_images[:length]
    # shuffle(shuffledtrain)
    # shuffle(shuffledvalidation)

    training_images = []

    for f in shuffledtrain:
        training_images.append(cv2.imread(f))
        y_train.append(0 if 'notfinish' in f else 1)

    for f in shuffledvalidation:
        test_images.append(cv2.imread(f))
        y_test.append(0 if 'notfinish' in f else 1)
        
    return training_images, y_train, test_images, y_test

def rescale(training_images, test_images):
    for i in tqdm(range(0,len(training_images))):
        training_images[i] = transform.resize(training_images[i],(imgrows,imgcols,3), mode='constant')

    for i in tqdm(range(0,len(test_images))):
        test_images[i] = transform.resize(test_images[i],(imgrows,imgcols,3), mode='constant')

    return training_images, test_images

def read_and_rescale():
    training_images, y_train, test_images, y_test = read_images()
    training_images, test_images = rescale(training_images, test_images)
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

    for i in y_train:
        print(i)

    y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test - y_test.min())
    return X_train, y_train

def train(X_train, y_train):
    adam = tf.keras.optimizers.Adam(learning_rate = 0.00001)

    # Neural network parameters
    #-----------------------------------------------
    #-----------------------------------------------
    batch_size = 64 # 
    epochs = 200 # 
    kernel_size=(3,3)
    #-----------------------------------------------
    #-----------------------------------------------
    num_classes = 2
    input_shape = (imgrows, imgcols, 3)
    print(input_shape)

    modelVGG19 = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=input_shape)

    type(modelVGG19)
    model = Sequential()

    model.add(tf.keras.layers.RandomZoom(height_factor=(-0.3, -0)))
    model.add(tf.keras.layers.RandomContrast(0.4))

    for layer in modelVGG19.layers[:]:
        model.add(layer)
    
    for layer in model.layers:
        layer.trainable = False
    
    model.add(Flatten()) 
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(num_classes,activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer =adam,metrics=['accuracy'])

    # model.summary()

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks = [tensorboard])
    model.save('./models/vgg19finishdetector256')
    return model

def test(model, X_test, y_test):
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