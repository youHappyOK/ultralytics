import cv2

from ultralytics import YOLO

from ultralytics.utils import LOGGER, ops

# Load a model
model = YOLO(r'C:\Users\Administrator\PycharmProjects\ultralytics\runs\detect\train3\weights\best.pt')  # pretrained YOLOv8n model

# conf：置信度阈值0.4，只展示0.4以上的目标框图
# iou：iou越低，就不会存在重叠的框图
results = model(r'D:\labelimg\test_plane2\images\train\3.png', conf=0.8, iou=0.1)



res = results[0].plot()
cv2.imshow("YOLOv8 Inference", res)
cv2.waitKey(0)