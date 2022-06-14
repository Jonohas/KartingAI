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

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Pipeline:
	def __init__(self):
		self.model = tf.keras.models.load_model('models/vgg19finishdetector256')
		self.values = []
		self.imgrows = 144
		self.imgcols = 256

		self.rescale_queue = Queue()
		self.prediction_queue = Queue()
		self.write_queue = Queue()
	
	@jit
	def read_video(self, video_url, destination_url):
		count = 0
		vidcap = cv2.VideoCapture(video_url)
		totalframecount= int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
		succes,image = vidcap.read()
		while succes:
			cv2.imwrite(f"{destination_url}frame%d.jpg" % count, image)
			succes, image = vidcap.read()
			print('Read a new frame: ', succes, f"{count}/{totalframecount}")
			count += 1

	@jit
	def read_images(self, dir):
		source = "./Frames/"
		print('Reading files...')

		batch_size = 300
		
		for root, dirs, files in walk(dir):
			print(len(files))
			batches = round(len(files) / batch_size)			
			for batch in tqdm(range(batches), desc="batches"):
				images = []

				for image in files[ batch_size * batch : batch_size * (batch+1)]:
					images.append( cv2.imread( path.join( root, image ) ) )

				self.rescale_queue.put(images)

			images = []
			for image in files[(batch_size*batches)-1:]:
				images.append( cv2.imread( path.join( root,image ) ) )

			self.rescale_queue.put(images)

	@jit
	def rescale(self):
		while True:
			if self.rescale_queue.empty():
				continue
			task = self.rescale_queue.get()
			new_files = []
			for i in tqdm(range(0,len(task)-1), desc="image rescaler"):
				new_files.append(cv2.resize(task[i], (self.imgcols, self.imgrows), interpolation=cv2.INTER_LINEAR))

			new_files = np.asarray(new_files, dtype="float32")
			new_files /= 255

			# test_images = np.asarray(files)
			new_files = new_files.reshape((len(new_files),self.imgrows,self.imgcols,3))

			self.prediction_queue.put([task, new_files])

	@jit
	def predict(self):
		while True:
			if self.prediction_queue.empty():
				continue
			
			task = self.prediction_queue.get()
			original_images = task[0]
			scaled_images = task[1]
			new_predictions = []
			predictions = self.model.predict(scaled_images)
			for prediction in predictions:
				if prediction[0] > prediction[1]:
					new_predictions.append(0)
				else:
					new_predictions.append(1)
			
			self.write_queue.put([original_images, new_predictions])

	@jit
	def write_image(self):
		w = 2704
		h = 1520
		fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
		writer = cv2.VideoWriter("written.mp4", fourcc, 60, (w, h))

		while True:
			if self.write_queue.empty():
				continue

			task = self.write_queue.get()
			images = task[0]
			predictions = task[1]
			for index in tqdm(range(len(predictions)), desc="image writer"):
				image = images[index]
				prediction = predictions[index]
				text = "finish" if prediction == 1 else "notfinish"
				color = (0,255,0) if prediction == 1 else (0,0,255)

				cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, color, 2, cv2.LINE_4)
				writer.write(image)


if __name__ == "__main__":
	os.remove("written.mp4")
	pl = Pipeline()
	# pl.read_video('Sequence 01.mp4', 'Frames/')
	threading.Thread(target=pl.rescale, daemon=True).start()
	threading.Thread(target=pl.predict, daemon=True).start()
	threading.Thread(target=pl.write_image, daemon=True).start()
	files = pl.read_images('./Frames/')
# rescaled = pl.rescale(files)

# pl.predict(rescaled, './Frames/', 'written.mp4')

