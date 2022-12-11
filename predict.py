import cv2
from os import walk, path, environ
import tensorflow as tf
from skimage import data, color, io, filters, morphology,transform, exposure, feature, util
import numpy as np
from tqdm import tqdm
import time
from queue import Queue
import threading
import os
from numba import jit

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_memory_growth(gpus[0], True)

class Pipeline:

    def __init__(self):
        model_path = ''
        for root, dirs, files in walk('output/models/current/'):
            model_path = path.join(root, dirs[0])


        self.model = tf.keras.models.load_model(model_path)

        self.imgrows = 54
        self.imgcols = 96

        self.batchsize = 2000

        self.images = []
        self.rescaled = []

        self.cap = cv2.VideoCapture('Source/cuts.mp4')
        # self.cap = cv2.VideoCapture('/home/jonas/Videos/24-10-2022/GX010493.MP4')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = cv2.VideoWriter('output.avi', fourcc, 59.94, (int(self.cap.get(3)),int(self.cap.get(4))), isColor=True)


    def read_images(self):
        print('Reading files...')

        length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        success, image = self.cap.read()
        count = 0


        batches = round(length / self.batchsize)
        batch = 0
        while success:
            # if batch is at max size
            start = batch * self.batchsize
            end = start + self.batchsize - 1
            self.images.append(image)

            print(f'image read {count} of {length} - {start} to {end}')
            if length < end and count == (length - 1):
                print(f'Batch {batch} of {batches} - {start} to {end}')
                self.rescale()
                self.predict()
                self.rescaled = []
                self.images = []
                batch += 1
            else:
                if count == end:
                    print(f'Batch {batch} of {batches} - {start} to {end}')
                    self.rescale()
                    self.predict()
                    self.rescaled = []
                    self.images = []
                    batch += 1

            count += 1

            success,image = self.cap.read()
        print('Read {} images'.format(count))
        self.cap.release()
        self.out.release()



    def rescale(self):
        for image in tqdm(self.images, desc="Rescaling"):
            shape = image.shape
            # resized_image = cv2.resize(image, (self.imgcols, self.imgrows), interpolation=cv2.INTER_LINEAR)
            image = tf.image.resize(image, (self.imgrows, self.imgcols))
            image = np.expand_dims(image, axis=0)
            self.rescaled.append(image)

    def predict(self):
        count = 0



        for image in tqdm(self.rescaled, desc="Predicting"):
            result = self.model.predict(image, batch_size = 0, verbose = 0)
            string = ''
            if result[0][0] == 1:
                string += "Not Finish"
            else:
                string += "Finish"
                

            fullscale_image = self.images[count]
            cv2.putText(
                fullscale_image, 
                string, 
                (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)

            # cv2.imshow('frame', image)
            # cv2.waitKey(0)

            self.out.write(fullscale_image)
            

            count += 1

    def run(self):
        self.read_images()
        # self.rescale()
        # self.predict()

if __name__ == "__main__":
    try:
        os.remove("output.avi")
    except:
        print('destination file doesnt exist, moving on')
        
    p = Pipeline()
    p.run()


    