import os

# 使用 os.system() 运行命令行命令
os.system("yolo detect train data=/app/data.yaml model=/app/yolov10-main/ultralytics/cfg/models/v10/yolov10s.yaml epochs=50 batch=8 imgsz=640 device=0")
