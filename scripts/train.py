
import argparse
import os
from glob import glob
import random
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy

from azureml.core import Run

parser = argparse.ArgumentParser()
parser.add_argument('--training-folder', type=str, dest='training_folder', help='training folder mounting point')
parser.add_argument('--testing-folder', type=str, dest='testing_folder', help='testing folder mounting point')
parser.add_argument('--epochs', type=int, dest='epochs', help='The amount of Epochs to train')
parser.add_argument('--batch-size', type=int, dest='batch_size', help='The batch size to use for training')
parser.add_argument('--learning-rate', type=float, dest='learning_rate', help='The learning rate to use for training')
parser.add_argument('--patience', type=int, dest='patience', help='The patience to use for training')
parser.add_argument('--model-name', type=str, dest='model_name', help='The model name to use for training')
parser.add_argument('--seed', type=int, dest='seed', help='The seed to use for training')
args = parser.parse_args()


# ================== IMPORT DATA ==================
# import data


imgrows, imgcols = 144, 254

X_train = []
y_train = []

X_test = []
y_test = []


for root, dirs, files in os.walk(args.training_folder):
    for f in files:
        X_train.append(cv2.imread(os.path.join(root,f)))
        if 'notfinish' in f:
            y_train.append(0)
        elif 'finish' in f:
            y_train.append(1)

for root, dirs, files in os.walk(args.testing_folder):
    for f in files:
        X_test.append(cv2.imread(os.path.join(root,f)))
        if 'notfinish' in f:
            y_test.append(0)
        elif 'finish' in f:
            y_test.append(1)


X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# for i in y_train:
#     print(i)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# ================== IMPORT DATA ==================








MAX_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
PATIENCE = args.patience
model_name = args.model_name
SEED = args.seed

# Create an output directory where our AI model will be saved to.
# Everything inside the `outputs` directory will be logged and kept aside for later usage.
model_path = os.path.join('outputs', model_name)
os.makedirs(model_path, exist_ok=True)

## START OUR RUN context.
## We can now log interesting information to Azure, by using these methods.
run = Run.get_context()

# Save the best model, not the last
cb_save_best_model = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                         monitor='val_loss', 
                                                         save_best_only=True, 
                                                         verbose=1)

# Early stop when the val_los isn't improving for PATIENCE epochs
cb_early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                              patience= PATIENCE,
                                              verbose=1,
                                              restore_best_weights=True)

# Reduce the Learning Rate when not learning more for 4 epochs.
cb_reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(factor=.5, patience=4, verbose=1)

# build model

adam = keras.optimizers.Adam(learning_rate = LEARNING_RATE)

num_classes = 2
input_shape = (imgrows, imgcols, 3)
print(input_shape)

modelVGG19 = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=input_shape)

type(modelVGG19)
model = Sequential()

for layer in model.layers[:]:
    model.add(layer)

for layer in modelVGG19.layers:
    layer.trainable = False

model.add(Flatten()) 
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(num_classes,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer =adam,metrics=['accuracy'])

# Construct & initialize the image data generator for data augmentation
# Image augmentation allows us to construct “additional” training data from our existing training data 
# by randomly rotating, shifting, shearing, zooming, and flipping. This is to avoid overfitting.
# It also allows us to fit AI models using a Generator, so we don't need to capture the whole dataset in memory at once.
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# train the network
history = model.fit_generator( aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
                        validation_data=(X_test, y_test),
                        steps_per_epoch=len(X_train) // BATCH_SIZE,
                        epochs=MAX_EPOCHS,
                        callbacks=[cb_save_best_model, cb_early_stop, cb_reduce_lr_on_plateau] )


print("[INFO] evaluating network...")
predictions = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=['notfinish', 'finish'])) # Give the target names to easier refer to them.
# If you want, you can enter the target names as a parameter as well, in case you ever adapt your AI model to more animals.

cf_matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
print(cf_matrix)

### TODO for students
### Find a way to log more information to the Run context.

# Save the confusion matrix to the outputs.
np.save('outputs/confusion_matrix.npy', cf_matrix)

print("DONE TRAINING")
