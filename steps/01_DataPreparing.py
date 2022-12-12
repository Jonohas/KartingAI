# Importing the default packages for data processing and visualisation
import numpy as np # Used to process our images in a data-format
import cv2 # Process the images


import os
from glob import glob
import math

import shutil

import warnings
warnings.filterwarnings("ignore") # Warnings that can be ignored will be ignored

import random



# Import AzureML packages
from azureml.core import Workspace
from azureml.core import Dataset
from azureml.data.datapath import DataPath


## Either get environment variables, or a fallback name, which is the second parameter.
## Currently, fill in the fallback values. Later on, we will make sure to work with Environment values. So we're already preparing for it in here!
workspace_name = os.environ.get('WORKSPACE', 'mlops-jonasfaber-karting')
subscription_id = os.environ.get('SUBSCRIPTION_ID', '7c50f9c3-289b-4ae0-a075-08784b3b9042')
resource_group = os.environ.get('RESOURCE_GROUP', 'NathanReserve')

LABELS = os.environ.get('LABELS', 'finish,notfinish').split(',')
SEED = int(os.environ.get('RANDOM_SEED', '42'))
TRAIN_TEST_SPLIT_FACTOR = float(os.environ.get('TRAIN_TEST_SPLIT_FACTOR', '0.2'))

# Connect to workspace
ws = Workspace.get(name=workspace_name,
               subscription_id=subscription_id,
               resource_group=resource_group)


def processAndUploadAnimalImages(datasets, data_path, processed_path, ws, label):
    label_path = os.path.join(data_path, 'karting', label)
    os.makedirs(label_path, exist_ok=True)

    # Get the dataset name for this animal, then download to the directory
    datasets[label].download(label_path, overwrite=True) # Overwriting means we don't have to delete if they already exist, in case something goes wrong.
    print('Downloading all the images')

        # Get all the image paths with the `glob()` method.
    print(f'Resizing all images for {label} ...')
    image_paths = glob(f"{label}/*.jpg") # CHANGE THIS LINE IF YOU NEED TO GET YOUR ANIMAL_NAMES IN THERE IF NEEDED!
    
    # Process all the images with OpenCV. Reading them, then resizing them to 64x64 and saving them once more.
    print(f"Processing {len(image_paths)} images")
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (254, 144)) # Resize to a square of 64, 64
        cv2.imwrite(os.path.join(processed_path, label, image_path.split('/')[-1]), image)
    print(f'... done resizing. Stopping context now...')


    # Upload the directory as a new dataset
    print(f'Uploading directory now ...')
    resized_dataset = Dataset.File.upload_directory(
                        # Enter the sourece directory on our machine where the resized pictures are
                        src_dir = os.path.join(processed_path, label),
                        # Create a DataPath reference where to store our images to. We'll use the default datastore for our workspace.
                        target = DataPath(datastore=ws.get_default_datastore(), path_on_datastore=f'resized_karting_images/{label}'),
                        overwrite=True)

    print('... uploaded images, now creating a dataset ...')

    # Make sure to register the dataset whenever everything is uploaded.
    new_dataset = resized_dataset.register(ws,
                            name=f'resized_{label}',
                            description=f'{label} images resized tot 254, 144',
                            tags={'labels': label, 'AI-Model': 'vgg19', 'GIT-SHA': os.environ.get('GIT_SHA')}, # Optional tags, can always be interesting to keep track of these!
                            create_new_version=True)

    print(f" ... Dataset id {new_dataset.id} | Dataset version {new_dataset.version}")
    print(f'... Done. Now freeing the space by deleting all the images, both original and processed.')
    emptyDirectory(label_path)
    print(f'... done with the original images ...')
    emptyDirectory(os.path.join(processed_path, label))
    print(f'... done with the processed images. On to the next Animal, if there are still!')

def emptyDirectory(directory_path):
    shutil.rmtree(directory_path)

def trainTestSplitData(ws):

    training_datapaths = []
    testing_datapaths = []
    default_datastore = ws.get_default_datastore()
    for label in LABELS:
        # Get the dataset by name
        karting_dataset = Dataset.get_by_name(ws, f"resized_{label}")
        print(f'Starting to process {label} images.')

        # Get only the .JPG images
        karting_images = [img for img in karting_dataset.to_path() if img.split('.')[-1] == 'jpg']

        print(f'... there are about {len(karting_images)} images to process.')

        ## Concatenate the names for the animal_name and the img_path. Don't put a / between, because the img_path already contains that
        karting_images = [(default_datastore, f'resized_karting_images/{label}{img_path}') for img_path in karting_images] # Make sure the paths are actual DataPaths
        
        random.seed(SEED) # Use the same random seed as I use and defined in the earlier cells
        random.shuffle(karting_images) # Shuffle the data so it's randomized
        
        ## Testing images
        amount_of_test_images = math.ceil(len(karting_images) * TRAIN_TEST_SPLIT_FACTOR) # Get a small percentage of testing images

        animal_test_images = karting_images[:amount_of_test_images]
        animal_training_images = karting_images[amount_of_test_images:]
        
        # Add them all to the other ones
        testing_datapaths.extend(animal_test_images)
        training_datapaths.extend(animal_training_images)

        print(f'We already have {len(testing_datapaths)} testing images and {len(training_datapaths)} training images, on to process more animals if necessary!')

    training_dataset = Dataset.File.from_files(path=training_datapaths)
    testing_dataset = Dataset.File.from_files(path=testing_datapaths)

    training_dataset = training_dataset.register(ws,
        name=os.environ.get('TRAIN_SET_NAME', 'karting-training-set'), # Get from the environment
        description=f'The Animal Images to train, resized tot 254, 144',
        tags={'labels': os.environ.get('LABELS'), 'AI-Model': 'vgg19', 'Split size': str(1 - TRAIN_TEST_SPLIT_FACTOR), 'type': 'training', 'GIT-SHA': os.environ.get('GIT_SHA')},
        create_new_version=True)

    print(f"Training dataset registered: {training_dataset.id} -- {training_dataset.version}")

    testing_dataset = testing_dataset.register(ws,
        name=os.environ.get('TEST_SET_NAME', 'karting-testing-set'), # Get from the environment
        description=f'The Animal Images to test, resized tot 254, 144',
        tags={'labels': os.environ.get('LABELS'), 'AI-Model': 'vgg19', 'Split size': str(TRAIN_TEST_SPLIT_FACTOR), 'type': 'testing', 'GIT-SHA': os.environ.get('GIT_SHA')},
        create_new_version=True)

    print(f"Testing dataset registered: {testing_dataset.id} -- {testing_dataset.version}")


data_folder = os.path.join(os.getcwd(), 'data')
os.makedirs(data_folder, exist_ok=True)
for label in LABELS:
    os.makedirs(os.path.join(data_folder, 'labels', label), exist_ok=True)

processed_path = os.path.join(os.getcwd(), 'data', 'processed', 'labels')
os.makedirs(processed_path, exist_ok=True)
for label in LABELS:
    os.makedirs(os.path.join(processed_path, label), exist_ok=True)

datasets = Dataset.get_all(workspace=ws) # Make sure to give our workspace with it
for label in LABELS:
    processAndUploadAnimalImages(datasets, data_folder, processed_path, ws, label)


trainTestSplitData(ws)

