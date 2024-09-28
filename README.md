# YOLOv10 Face Detection

本專案使用 YOLOv10 進行人臉偵測，並包含模型訓練、驗證和預測的完整流程。

## 安裝

建議使用 conda 虛擬環境。

```sh
conda create -n yolov10 python=3.9
conda activate yolov10
pip install -r requirements.txt
pip install -e .
```

## 訓練

使用以下命令進行模型訓練：

```sh
yolo detect train data=coco.yaml model=yolov10n/s/m/b/l/x.yaml epochs=500 batch=256 imgsz=640 device=0,1,2,3,4,5,6,7
```

或使用 Python 代碼：

```python
from ultralytics import YOLOv10

model = YOLOv10()
model.train(data='coco.yaml', epochs=500, batch=256, imgsz=640)
```

## 驗證

使用預訓練模型進行驗證：

```sh
yolo val model=jameslahm/yolov10{n/s/m/b/l/x} data=coco.yaml batch=256
```

或使用 Python 代碼：

```python
from ultralytics import YOLOv10

model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
model.val(data='coco.yaml', batch=256)
```


## 預測

使用以下命令進行預測：

```sh
yolo predict model=jameslahm/yolov10{n/s/m/b/l/x}
```

或使用 Python 代碼：

```python
from ultralytics import YOLOv10

model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
model.predict()
```

## 匯出

將模型匯出為 ONNX 或 TensorRT 格式：

```sh
# 匯出為 ONNX
yolo export model=jameslahm/yolov10{n/s/m/b/l/x} format=onnx opset=13 simplify
# 使用 ONNX 進行預測
yolo predict model=yolov10n/s/m/b/l/x.onnx

# 匯出為 TensorRT
yolo export model=jameslahm/yolov10{n/s/m/b/l/x} format=engine half=True simplify opset=13 workspace=16
# 或
trtexec --onnx=yolov10n/s/m/b/l/x.onnx --saveEngine=yolov10n/s/m/b/l/x.engine --fp16
# 使用 TensorRT 進行預測
yolo predict model=yolov10n/s/m/b/l/x.engine
```

或使用 Python 代碼：

```python
from ultralytics import YOLOv10

model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
model.export(...)
```

## 致謝

本專案基於 [ultralytics](https://github.com/ultralytics/ultralytics) 和 [RT-DETR](https://github.com/lyuwenyu/RT-DETR) 的實現。

感謝這些優秀的實現！

## 引用

```BibTeX
@article{wang2024yolov10,
  title={YOLOv10: Real-Time End-to-End Object Detection},
  author={Wang, Ao and Chen, Hui and Liu, Lihao and Chen, Kai and Lin, Zijia and Han, Jungong and Ding, Guiguang},
  journal={arXiv preprint arXiv:2405.14458},
  year={2024}
}
```
