from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')
model.info()

if __name__ == '__main__':

    # Train the model
    results = model.train(data=r'C:\Users\Administrator\PycharmProjects\ultralytics\ultralytics\cfg\datasets\airplane.yaml', epochs=500, imgsz=640)
    # 指定gpu训练
    # results = model.train(data=r'C:\Users\Administrator\PycharmProjects\ultralytics\ultralytics\cfg\datasets\airplane.yaml', epochs=50, imgsz=640, device=0)