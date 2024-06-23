from ultralytics import YOLO
from os import path
CUDA_LAUNCH_BLOCKING=1


#load model
model = YOLO('yolov8n.pt')
model.info()  # display info



results = model.train(data='./src/data.yaml', epochs=500, batch=8)