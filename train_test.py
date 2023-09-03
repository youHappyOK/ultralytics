from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
# 首先导入了YOLO类，并使用预训练的检测模型yolov8n.pt模型文件创建了一个model对象。model对象是使用YOLO类初始化得到的YOLOv8模型。
model = YOLO('yolov8n.pt')
# model.to('cuda')
# Display model information (optional)
model.info()
if __name__ == '__main__':
    # 这段代码调用了model对象的train方法来对模型进行训练。其中：
    # data='coco8.yaml'：指定了数据集配置文件的路径，该文件包含了数据集的相关参数和路径信息。
    # epochs=10：指定了训练的总轮数（即迭代次数），在这个例子中为100，表示将整个数据集的样本都用于训练10次。
    # imgsz=640：指定了输入图像的大小，这里为640x640像素。大多数目标检测模型在训练过程中需要将输入图像缩放到固定的尺寸
    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data='coco128.yaml', epochs=5, imgsz=640)

    # 这段代码使用训练完成的YOLOv8模型对指定的bus.jpg图像进行目标检测。检测结果会存储在results中，你可以根据model对象的具体实现来查看和使用这些结果。
    # Run inference with the YOLOv8n model on the 'bus.jpg' image
    # results = model('path/to/bus.jpg')