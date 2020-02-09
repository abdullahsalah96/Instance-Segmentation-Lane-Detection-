from helper import Segmentation
import cv2
import json
import numpy as np
from sklearn.datasets import load_files

IMAGES_PATH = "/Users/abdallaelshikh/Desktop/College/Graduation Project/tuSimple Dataset/train_set (1)/clips/0313-1/60"
JSON_PATH = "/Users/abdallaelshikh/Desktop/College/Graduation Project/tuSimple Dataset/train_set (1)/label_data_0601.json"
FOLDER_PATH = "/Users/abdallaelshikh/Desktop/College/Graduation Project/tuSimple Dataset/train_set (1)/"
OUTPUT_ANNOTATIONS_PATH = "/Users/abdallaelshikh/Desktop/College/Graduation Project/tuSimple Dataset/train_set (1)/annotations/"
OUTPUT_IMAGES_PATH = "/Users/abdallaelshikh/Desktop/College/Graduation Project/tuSimple Dataset/train_set (1)/images/"
INDEX = 1713

s = Segmentation()

def generate_labels(JSON_PATH, FOLDER_PATH, OUTPUT_PATH, index):
    json_gt = [json.loads(line) for line in open(JSON_PATH)]
    print(len(json_gt))
    for sample_num in range(len(json_gt)):
        gt = json_gt[sample_num]
        gt_lanes = gt['lanes']
        y_samples = gt['h_samples']
        raw_file = gt['raw_file']
        print("Generating Label " + str(raw_file))
        img = cv2.imread(FOLDER_PATH + raw_file)
        gt_lanes_vis = [[(x,y) for (x,y) in zip(lane, y_samples) if x>=0] for lane in gt_lanes]

        mask = np.zeros_like(img)
        colors = [[255,0,0], [0,255,0], [0,0,255], [0,255,255], [255,255,0],[255,0,255]]

        for i in range(len(gt_lanes_vis)):
            cv2.polylines(mask, np.int32([gt_lanes_vis[i]]), isClosed = False, color=colors[i], thickness = 5)

        image_label = np.zeros_like(img, dtype = np.uint8)
        for i in range(4):
            image_label[np.where((mask == colors[i]).all(axis=2))] = i+1
        output_path = OUTPUT_PATH + str(sample_num + index) + ".png"
        cv2.imwrite(output_path, image_label)
        
        print(str(len(json_gt) - sample_num) + " samples left")


def generate_images(JSON_PATH, FOLDER_PATH, OUTPUT_PATH, index):
    json_gt = [json.loads(line) for line in open(JSON_PATH)]
    print(len(json_gt))
    for sample_num in range(len(json_gt)):
        gt = json_gt[sample_num]
        gt_lanes = gt['lanes']
        y_samples = gt['h_samples']
        raw_file = gt['raw_file']
        print("Generating Images " + str(raw_file))
        img = cv2.imread(FOLDER_PATH + raw_file)
        
        output_path = OUTPUT_PATH + str(sample_num + index) + ".png"
        cv2.imwrite(output_path, img)
        print(str(len(json_gt) - sample_num) + " samples left")



# generate_labels(JSON_PATH, FOLDER_PATH, OUTPUT_ANNOTATIONS_PATH, INDEX)
generate_images(JSON_PATH, FOLDER_PATH, OUTPUT_IMAGES_PATH, INDEX)