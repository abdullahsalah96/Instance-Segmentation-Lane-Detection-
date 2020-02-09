from keras.models import load_model
from helper import Segmentation
import numpy as np
import cv2
from config import *

im = Segmentation()

def predict_class(model, img_path):
    """
    A funtion that takes the model and the path of the image to be predicted and returns the prediction
    """
    image = im.path_to_tensor(img_path, True , (128,128), GRAYSCALE)
    prediction = model.predict(image)
    return np.reshape(prediction,(128,128,NUM_OF_CLASSES))

PATH = "/Users/abdallaelshikh/Desktop/College/Graduation Project/Instance-Segmentation-Lane-Detection-/Dataset/Testing/images/image1000.png"

loaded_model = load_model("lane_model.h5")

img = cv2.imread(PATH)
prediction = predict_class(loaded_model, PATH)


print(prediction[:,:,1])
out = np.zeros((128,128,1), dtype = np.uint8)

segmentation_map = np.zeros((128,128,3), np.uint8)
#getting the maximum pixel value of the one hot encoded array
for i in range(128):
    for j in range(128):
        out[i][j] = np.argmax(prediction[i][j])
        if(out[i][j] == 0):
            segmentation_map[i,j] = (0,50,0)
        elif(out[i][j] == 1):
            segmentation_map[i,j] = (0,0,255)
        elif(out[i][j] == 2):
            segmentation_map[i,j] = (255,0,0)
        elif(out[i][j] == 3):
            segmentation_map[i,j] = (255,0,255)
        elif(out[i][j] == 4):
            segmentation_map[i,j] = (0,255,255)
        elif(out[i][j] == 5):
            segmentation_map[i,j] = (255,255,0)


src = img
src = cv2.resize(src, (480,360))
out = cv2.resize(out, (480,360))
segmentation_map = cv2.resize(segmentation_map, (480,360))
vis_image = cv2.addWeighted(src,1.0,segmentation_map,0.8,0)
cv2.imshow('src', src)
cv2.imshow('map', segmentation_map)
cv2.imshow('visualization', vis_image)
cv2.waitKey(0)