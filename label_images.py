import glob
import cv2

# pip install pillow
from PIL import Image, ImageTk


count = 0
for file in glob.glob('Frames/*.jpg'):
    
    img = cv2.imread(file)
    cv2.imshow(file, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    online = input("Is this driver on the finish line? [Y/n]")

    while online.lower() not in 'yn':
        online = input("Something went wrong. Is this driver on the finish line? [Y/n]")

    if online == 'y' or online == 'Y':
        # driver is on finish line
        new_filename = f"finish_{count}"
    else:
        # driver is not on finish line
        new_filename = f"notfinish_{count}"

    cv2.imwrite(f'Labeled/{new_filename}.jpg',img)

    count += 1
    