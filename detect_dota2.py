import numpy as np
import win32gui
from PIL import ImageGrab

from ultralytics import YOLO
import cv2
# 加载预训练模型
model = YOLO(r"C:\Users\Administrator\PycharmProjects\ultralytics\runs\detect\train\weights\best.pt", task='detect')
model.to('cuda')
# model = YOLO("yolov8n.pt") task参数也可以不填写，它会根据模型去识别相应任务类别
# 检测图片
# results = model("./ultralytics/assets/img_4.png")
# res = results[0].plot()
# cv2.imshow("YOLOv8 Inference", res)
# cv2.waitKey(0)
rect = win32gui.GetWindowRect(132770)
window_rect = (rect[0], rect[1], rect[2], rect[3])
while True:
    dota2Image = ImageGrab.grab(window_rect)
    npImg = np.array(dota2Image)
    # 因为pillow的图像是rgb的，所以要转成cv2需要的bgr
    dota2Image = cv2.cvtColor(npImg, cv2.COLOR_RGB2BGR)
    # 检测图片
    results = model(dota2Image)
    res = results[0].plot()
    cv2.imshow("YOLOv8 Inference", res)
    # 检查按键事件
    cv2.waitKey(1)