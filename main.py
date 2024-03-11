import cv2
import numpy as np
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from Utils import *

imgDir = r'D:\traffic_sign_detection\images'
annotationDir = r'D:\traffic_sign_detection\annotations'

imgLst = []
labelLst = []

for xmlFile in tqdm(os.listdir(annotationDir)):
    xml = os.path.join(annotationDir,xmlFile)
    tree = ET.parse(xml)
    root = tree.getroot()

    folder = root.find('folder').text
    imgName = root.find('filename').text
    imgFileDir = os.path.join(imgDir,imgName)
    img = cv2.imread(imgFileDir)
    
    for obj in root.findall('object'):
        classname = obj.find('name').text
        if classname == 'trafficlight':
            continue
        else:
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)

            objectImg = img[ymin:ymax,xmin:xmax]
            imgLst.append(objectImg)
            labelLst.append(classname)
print(f'Number object: {len(imgLst)}')
print(f'Class names: {list(set(labelLst))}')

imgFeatureLst = []
for img in imgLst:
    hogFeature = preprocessImg(img)
    imgFeatureLst.append(hogFeature)
imgFeatures = np.array(imgFeatureLst)

print(f'Image shape: {imgLst[0].shape}')
print(f'Features shape: {imgFeatures[0].shape}')

labelEncoder = LabelEncoder()
encoderLabels = labelEncoder.fit_transform(labelLst)

randomState = 0
testSize = 0.3
isShuffle = True

Xtrain,Xval,ytrain,yval = train_test_split(imgFeatures,encoderLabels,
                                           test_size=testSize,
                                           random_state=randomState,
                                           shuffle=isShuffle)

scaler = StandardScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xval = scaler.transform(Xval)

clf = SVC(kernel='rbf',
          random_state=randomState,
          probability=True,
          C=0.5)
clf.fit(Xtrain,ytrain)

ypred = clf.predict(Xval)
score = accuracy_score(ypred,yval)

print(f'Accuracy score: {score}')

img = cv2.imread(r'D:\traffic_sign_detection\images\road870.png')
predict(clf,scaler,img,labelEncoder)
