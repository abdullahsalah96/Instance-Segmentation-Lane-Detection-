from helper import Segmentation
import cv2
import json
import numpy as np
from sklearn.datasets import load_files

IMAGES_PATH = "/Users/abdallaelshikh/Desktop/College/Graduation Project/tuSimple Dataset/train_set (1)/Dataset/images/"
ANNOTATIONS_PATH = "/Users/abdallaelshikh/Desktop/College/Graduation Project/tuSimple Dataset/train_set (1)/Dataset/annotations/"

segmentation = Segmentation()


img = cv2.imread(IMAGES_PATH + "image0926.png")
annotation = cv2.imread(ANNOTATIONS_PATH + "image0926.png", cv2.IMREAD_GRAYSCALE)
out = annotation
segmentation_map = np.zeros((out.shape[0],out.shape[1],3), np.uint8)
print(out.shape)
for i in range(out.shape[0]):
    for j in range(out.shape[1]):
        if(out[i,j] == 0):
            segmentation_map[i,j] = (0,50,0)
        elif(out[i][j] == 1):
            segmentation_map[i,j] = (0,0,255)
        elif(out[i][j] == 2):
            segmentation_map[i,j] = (255,0,0)
        elif(out[i][j] == 3):
            segmentation_map[i,j] = (255,100,0)

vis_image = cv2.addWeighted(img,1.0,segmentation_map,0.8,0)
cv2.imshow("img", img)
cv2.imshow("mask", annotation)
cv2.imshow("vis", vis_image)
cv2.waitKey(0)