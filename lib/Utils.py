import cv2
import os
import numpy as np
from skimage.transform import resize
from skimage import feature
from sklearn.preprocessing import LabelEncoder,StandardScaler

class Config:
    confThreshold = 0.95
    stride = 12
    windowSize = [(32,32),(64,64),(128,128),(256,256),(512,512)]


def nms(bboxes, iouThreshold):
    bboxes.sort(key=lambda x: x[5], reverse=True)
    selected_bboxes = []
    while len(bboxes) > 0:
        max_conf_bbox = bboxes.pop(0)
        selected_bboxes.append(max_conf_bbox)
        iou_values = [calculate_iou(max_conf_bbox, bbox) for bbox in bboxes]
        filtered_bboxes = [bbox for i, bbox in enumerate(bboxes) if iou_values[i] <= iouThreshold]
        bboxes = filtered_bboxes
    return selected_bboxes

def calculate_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
    intersection_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    union_area = w1 * h1 + w2 * h2 - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def preprocessImg(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    resizeImg = resize(img,output_shape=(32,32),anti_aliasing=True)
    hogFeature = feature.hog(resizeImg,
                             orientations=9,
                             pixels_per_cell=(8,8),
                             cells_per_block=(2,2),
                             transform_sqrt=True,
                             block_norm='L2',
                             feature_vector=True)
    return hogFeature

def slidingWindow(img,windowSizes,stride):
    imgH,imgW = img.shape[:2]
    window = []
    for windowSize in windowSizes:
        windowW,windowH = windowSize
        for ymin in range(0,imgH-windowH+1,stride):
            for xmin in range(0,imgW-windowW+1,stride):
                xmax = xmin + windowW
                ymax = ymin + windowH
                window.append([xmin,ymin,xmax,ymax])
    return window

def pyramid(img,scale=0.8,minsize=(30,30)):
    accScale = 1.0
    pyramidImgs = [(img,accScale)]
    
    while True:
        accScale = accScale * scale
        h = int(img.shape[0] * accScale)
        w = int(img.shape[1] * accScale)
        if h < minsize[1] or w < minsize[0]:
            break
        img = cv2.resize(img,(w,h))
        pyramidImgs.append((img,accScale))
    return pyramidImgs

def visualizeBbox(img,bbox,labelEncoder):
    for box in bbox:
        xmin,ymin,xmax,ymax,predictId,confScore = box
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,255),2)
        classname = labelEncoder.inverse_transform([predictId])[0]
        label = f'{classname} {confScore:.2f}'
        (w,h),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_COMPLEX,0.6,1)
        cv2.rectangle(img,(xmin,ymin-20),(xmin+w,ymin),(0,255,0),-1)
        cv2.putText(img,label,(xmin,ymin-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),1)
    if bbox != []:
        # imgDir = r'D:\traffic_sign_detection\predict\result.jpg'
        # cv2.imwrite(imgDir,img)
        cv2.imshow('image',img)
        cv2.waitKey(0)

def predict(clf,scaler,image,labelEncoder):
    bbox = []
    pyramidImgs = pyramid(image)
    for pyramidImgInfo in pyramidImgs:
        pyramidImg,scaleFactor = pyramidImgInfo
        windowLst = slidingWindow(pyramidImg,Config.windowSize,Config.stride)
        for window in windowLst:
            xmin,ymin,xmax,ymax = window
            objectImg = pyramidImg[ymin:ymax,xmin:xmax]
            preprocessImage = preprocessImg(objectImg)
            normalizeImg = scaler.transform([preprocessImage])[0]
            decision = clf.predict_proba([normalizeImg])[0]
            if np.all(decision < Config.confThreshold):
                continue
            else:
                predictId = np.argmax(decision)
                confScore = decision[predictId]
                xmin = int(xmin/scaleFactor)
                xmax = int(xmax/scaleFactor)
                ymin = int(ymin/scaleFactor)
                ymax = int(ymax/scaleFactor)
                bbox.append([xmin,ymin,xmax,ymax,predictId,confScore])
    bbox = nms(bbox,iouThreshold=0.01)
    visualizeBbox(image,bbox,labelEncoder)


