import numpy as np
from core.image_cv2 import pil_to_cv2
import cv2
import openvino as ov
import psutil
import time
import os
import sys

def get_resource_path(relative_path):
    """ Get the absolute path to a resource in a PyInstaller bundle. """
    # Check if running in a PyInstaller bundle
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)
#Modify part 
model_path_xml = get_resource_path('/home/hquanhh/data_plate/ai-engine-main/pretrained/plate/model_SDD.xml')
model_path_bin = get_resource_path('/home/hquanhh/data_plate/ai-engine-main/pretrained/plate/model_SDD.bin')

def draw_bounding_boxes(image_path, boxes, scores, classes, threshold=0.6):
    global saved_boxes
    saved_boxes = []  # Reset saved_boxes each time the function is called
    image = image_path
    for i in range(len(boxes)):
        if scores[i] > threshold and int(classes[i]) == 1:
            box = boxes[i]
            score = scores[i]
            y_min, x_min, y_max, x_max = box
            start_point = (int(x_min * image.shape[1]), int(y_min * image.shape[0]))
            end_point = (int(x_max * image.shape[1]), int(y_max * image.shape[0]))
            image = cv2.rectangle(image, start_point, end_point, (255, 0, 0), 2)
            label = f'Class: 1, Score: {score:.2f}'
            image = cv2.putText(image, label, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            saved_boxes.append({
                'box': [start_point, end_point],
                'class_id': 1,
                'score': score
            })
    img_cv2 = pil_to_cv2(image)
    return img_cv2,saved_boxes

def infer_and_visualize(model_path_xml, model_path_bin, image_path):
    saved_boxes = []
    core = ov.Core()
    model = core.read_model(model_path_xml, model_path_bin)
    compiled_model = core.compile_model(model, "CPU")
    image = image_path
    input_tensor = image

    input_tensor = np.expand_dims(input_tensor.astype(np.uint8), axis=0)

    cpu_before = psutil.cpu_percent(interval=None)
    start_time = time.time()
    output = compiled_model([input_tensor])
    end_time = time.time()
    cpu_after = psutil.cpu_percent(interval=None)

    inference_time = end_time - start_time
    print(f"Inference Time: {inference_time:.4f} seconds")
    cpu_usage_during_inference = (cpu_after + cpu_before) / 2
    print(f"Average CPU usage during inference: {cpu_usage_during_inference:.2f}%")

    detection_boxes = output[compiled_model.output(1)]
    detection_classes = output[compiled_model.output(2)]
    detection_scores = output[compiled_model.output(4)]
    
    img_cv2 ,coordinate= draw_bounding_boxes(image, detection_boxes[0], detection_scores[0], detection_classes[0])
    
    print(saved_boxes)
    return img_cv2, coordinate
