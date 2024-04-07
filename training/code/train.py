from ultralytics import YOLO
from os import path
CUDA_LAUNCH_BLOCKING=1


#load model
model = YOLO('./runs/detect/train6/weights/best.pt')
model.info()  # display info



results = model.train(data='./code/data.yaml', epochs=300, batch=8)