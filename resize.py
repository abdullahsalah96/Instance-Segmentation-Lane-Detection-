import cv2
import glob
import os

INPUT_PATH = "/Users/abdallaelshikh/Desktop/College/Graduation Project/tuSimple Dataset/train_set (1)/annotations"
OUTPUT_PATH = "/Users/abdallaelshikh/Desktop/College/Graduation Project/tuSimple Dataset/train_set (1)/Dataset/annotations"
RESIZE_FACTOR = 3
os.makedirs(OUTPUT_PATH, exist_ok=True)
image_num = 0

for img in sorted(glob.glob(INPUT_PATH + "/*.png")):
    print("RESIZING IMAGE " + str(image_num))
    image = cv2.imread(img)
    resized = cv2.resize(image,(int(image.shape[1]/RESIZE_FACTOR), int(image.shape[0]/RESIZE_FACTOR)))
    cv2.imwrite(OUTPUT_PATH + "/image%04i.png" %image_num, resized)
    image_num+=1

print("SAVED RESIZED IMAGES")

