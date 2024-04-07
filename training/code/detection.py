import cv2 as cv
from ultralytics import YOLO

#load model

model = YOLO('runs/detect/train7/weights/best.pt')

#detect
results = model.track(source="http://192.168.129.56:4747/video", show=True,conf=0.4)