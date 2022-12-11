# read video file with opencv

from platform import architecture
import cv2
from os import walk, path, environ
from matplotlib import test
from tqdm import tqdm
from PIL import ImageTk, Image
import threading
import sys

print('Starting...')

# create gui with tkinter
import tkinter as tk



lock = threading.Lock()

count = 0
location = ""
global_image = None
is_on_finish_line = False
next_image = False




def save_image():
    # save image to ./Source/labels
    try:
        new_filename = f"notfinish/notfinish_{location}"
        if (is_on_finish_line):
            new_filename = f"finish/finish_{location}"

        print(new_filename)
            
        cv2.imwrite(f"./Source/Labeled/{new_filename}", global_image)
    except Exception as e:
        print("Error saving image", e)

def submit():
    global next_image
    save_image()
    next_image = True

def on_line():
    global is_on_finish_line
    is_on_finish_line = True

def not_on_line():
    global is_on_finish_line
    is_on_finish_line = False

def test_thread(lock):
    global count
    with lock:
        count = 1000000
    sys.exit()


def read_images(lock, folder):
    global count, global_image, location, next_image, is_on_finish_line
    print('Reading files...')

    for base, dirs, files in walk(folder):
        for file in files:
            with lock:
                # read image    
                image_path = path.join( base, file )
                image = cv2.imread( image_path )

                I = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image_array = Image.fromarray(I)
                image_obj = ImageTk.PhotoImage(image_array)
                
                resized = image_array.resize((256,144))

                image_obj_res = ImageTk.PhotoImage(resized)

                canvas.create_image(20, 20, anchor=tk.NW, image=image_obj_res)
                canvas.image = image_obj_res
                location = "%d.jpg" % count
                global_image = image

                root.update()
                while not next_image:
                    pass

                next_image = False

                count += 1



def read_video(lock):
    global count, global_image, location, next_image, is_on_finish_line
    # read video from ./Source/video.mp4
    source = "./Source/cuts.mp4"
    try:
        vidcap = cv2.VideoCapture(source)
        totalframecount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        succes,image = vidcap.read()

        while succes:
            
                # cv2.imwrite(f"./Source/Frames/frame%d.jpg" % count, image)

                succes, image = vidcap.read()

                if (succes):

                    with lock:

                        I = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        image_array = Image.fromarray(I)
                        image_obj = ImageTk.PhotoImage(image_array)

                        resized = image_array.resize((256,144))
                        image_obj_res = ImageTk.PhotoImage(resized)

                        canvas.create_image(20, 20, anchor=tk.NW, image=image_obj_res)

                        canvas.image = image_obj_res
                        location = "%d.jpg" % count
                        print(location)
                        global_image = cv2.resize(image, (256,144))
                        
                        root.update()
                        while not next_image:
                            pass

                        next_image = False


                        count += 1

                        print(count, location)

    except Exception as e:
        print("Error reading video", e)

    finally:
        print("Finished reading video")
        sys.exit()


root = tk.Tk()
root.title("Video to frames")

# create a frame
app = tk.Frame(root)
app.grid()


canvas = tk.Canvas(root, width = 300, height = 300)
canvas.grid(columnspan=3, rowspan=3)

# create buttons
btn_ol = tk.Button(app, text="On the line", command=on_line)
btn_ol.grid(columnspan=3, rowspan=3)

btn_nol = tk.Button(app, text="Not on the line", command=not_on_line)
btn_nol.grid(columnspan=3, rowspan=3)

btn_nol = tk.Button(app, text="Submit", command=submit)
btn_nol.grid(columnspan=3, rowspan=3)
# read_video()

if __name__ == "__main__":
    x = threading.Thread(target=read_video, args=(lock,))
    y = threading.Thread(target=read_images, args=(lock,'Source/Frames/',))
    # create a button

    btn = tk.Button(app, text="Start video", command=x.start)
    btn.grid(columnspan=3, rowspan=3)

    btn = tk.Button(app, text="Start images", command=y.start)
    btn.grid(columnspan=3, rowspan=3)

    # start the gui
    root.mainloop()


    print(count)




# create close button in tkinter