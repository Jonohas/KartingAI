import cv2
from tqdm import tqdm
import os
import queue
import threading



def read_video_file(destination_path, q, resize_shape = (224, 224)):
    imgrows = resize_shape[0]
    imgcols = resize_shape[1]
    while True:
        if q.empty():
            break
        video_file = q.get()
        full_path = video_file[0]
        file_name = video_file[1]


        # read video file with opencv
        cap = cv2.VideoCapture(full_path)

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        success, image = cap.read()
        count = 0

        progress_bar = tqdm(total=length)
        while success:
            if success == False:
                break
            # resize image

            # show_image = ((count < 5000) or (count > (length - 5000)))

            success,image = cap.read()

            if file_name != "cuts.mp4":

                
                if (count % 160 == 0):
                    # resized = cv2.resize(image, (imgrows, imgcols))
                    cv2.imwrite(f"{destination_path}{file_name}{count}.jpg", image)     # save frame as JPEG file 
                    progress_bar.set_description_str(f"{file_name} {count}/{length}")

            else:
                cv2.imwrite(f"{destination_path}{file_name}{count}.jpg", image)     # save frame as JPEG file 
                progress_bar.set_description_str(f"{file_name} {count}/{length}")

            progress_bar.update(1)
            count += 1

        cap.release()  
        break
        

def read_videos_folder(source, q):
    for file in os.listdir(source):
        if file.endswith(".mp4"):
            print(f"Adding file to queue: {file}")
            q.put((f"{source}{file}", file))


if __name__ == "__main__":
    imgrows = 144
    imgcols = 256

    q = queue.Queue()
    read_videos_folder("Source/Videos/", q)

    threads = []

    for i in range(6):
        t = threading.Thread(target=read_video_file, args=("Source/Frames/", q, (imgcols, imgrows)))
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()

