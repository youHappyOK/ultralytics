import torch

from ultralytics import YOLO
import cv2
# 加载预训练模型
model = YOLO("yolov8n.pt", task='detect')

# 检查是否有可用的 GPU 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 在检测阶段使用gpu
model = model.to(device)

# model = YOLO("yolov8n.pt") task参数也可以不填写，它会根据模型去识别相应任务类别
# 检测图片
results = model("./ultralytics/assets/img_3.png")


res = results[0].plot()
cv2.imshow("YOLOv8 Inference", res)
cv2.waitKey(0)