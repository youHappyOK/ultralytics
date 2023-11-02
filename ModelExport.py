#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    @Author：吉姆哥儿
    @file： ModelExport.py
    @date：2023/11/1 11:37
    @desc: 
"""
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
model = YOLO(r'C:\Users\Administrator\PycharmProjects\ultralytics\runs\detect\train\weights\best.pt')  # load a custom trained model

# Export the model
model.export(format='onnx')
