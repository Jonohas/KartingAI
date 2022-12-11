import cv2
from os import walk, path, environ
from tqdm import tqdm

print('Starting...')

def read_images():
    # read images from ./Source/labels
    source = "./Source/Labeled"
    print('Reading files...')

    for root, dirs, files in walk(source):
        print(len(files))
        try:
            for file in files:
                image_path = path.join( root, file )
                image = cv2.imread( path.join( root, file ) )
                # show the image with cv2
                cv2.imshow('image', image)
                cv2.waitKey(0)
                
        except Exception as e:
            print("Error reading files", e)

    cv2.destroyAllWindows()


read_images()