import psutil
import cv2
import time
from core.crop_plate import crop_image
import core.pre_post_processing as processing
from core.image_process import prep_for_rec, batch_text_box, image_preprocess, post_processing_detection
from ultralytics import YOLO
import openvino as ov
import collections
from PIL import Image
import numpy as np
import re
from pathlib import Path

#Modify part

# Load model detection
det_model_file_path = Path("/home/hquanhh/data_plate/det/det2/inference.pdmodel")
# Initialize OpenVINO Runtime for text detection.
core = ov.Core()
det_model = core.read_model(det_model_file_path)
det_compiled_model = core.compile_model(det_model, device_name="AUTO")

# Get input and output nodes for text detection.
det_input_layer = det_compiled_model.input(0)
det_output_layer = det_compiled_model.output(0)

rec_model_file_path = Path("/home/hquanhh/data_plate/rec/inference.pdmodel")
# Read the model and corresponding weights from a file.
rec_model = core.read_model(model=rec_model_file_path)

# Assign dynamic shapes to every input layer on the last dimension.
for input_layer in rec_model.inputs:
    input_shape = input_layer.partial_shape
    input_shape[3] = -1
    rec_model.reshape({input_layer: input_shape})

rec_compiled_model = core.compile_model(model=rec_model, device_name="AUTO")

# Get input and output nodes.
rec_input_layer = rec_compiled_model.input(0)
rec_output_layer = rec_compiled_model.output(0)

def run_paddle_ocr_with_plate(image_path, saved_boxes, use_popup=False):
    # Initialize processing times and variables
    processing_times = collections.deque()
    txts = []
    scores = []
    final_text = ''  # Initialize final_text

    # Time tracking
    total_detection_time = 0
    total_recognition_time = 0
    total_detection_cpu = 0
    total_recognition_cpu = 0
    
    # Step 1: Crop images from the original image based on saved_boxes
    cropped_images, saved_boxes = crop_image(image_path)

    # Check if there are cropped images
    if not cropped_images:
        print("No cropped images found.")
        return final_text  # Return empty if no plates are detected

    for idx, plate_region in enumerate(cropped_images):
        # Preprocess the cropped image for text detection
        test_image = image_preprocess(plate_region, 640)

        ### Text Detection Phase ###
        cpu_before_detection = psutil.cpu_times_percent()
        start_time = time.time()

        # Perform detection inference
        det_results = det_compiled_model([test_image])[det_output_layer]

        stop_time = time.time()
        cpu_after_detection = psutil.cpu_times_percent()

        # Calculate detection time and CPU usage
        detection_time = stop_time - start_time
        total_detection_time += detection_time
        detection_cpu = cpu_after_detection.user - cpu_before_detection.user
        total_detection_cpu += detection_cpu

        # Postprocess detection results
        dt_boxes = post_processing_detection(plate_region, det_results)
        processing_times.append(detection_time)
        if len(processing_times) > 200:
            processing_times.popleft()
        processing_time_det = np.mean(processing_times) * 1000

        ### Text Recognition Phase ###
        dt_boxes = processing.sorted_boxes(dt_boxes)
        batch_num = 6
        img_crop_list, img_num, indices = prep_for_rec(dt_boxes, plate_region)

        rec_res = [['', 0.0]] * img_num

        for beg_img_no in range(0, img_num, batch_num):
            norm_img_batch = batch_text_box(img_crop_list, img_num, indices, beg_img_no, batch_num)

            cpu_before_recognition = psutil.cpu_times_percent()
            start_recognition_time = time.time()

            rec_results = rec_compiled_model([norm_img_batch])[rec_output_layer]

            stop_recognition_time = time.time()
            cpu_after_recognition = psutil.cpu_times_percent()

            recognition_time = stop_recognition_time - start_recognition_time
            total_recognition_time += recognition_time
            recognition_cpu = cpu_after_recognition.user - cpu_before_recognition.user
            total_recognition_cpu += recognition_cpu

            postprocess_op = processing.build_post_process(processing.postprocess_params)
            rec_result = postprocess_op(rec_results)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
            if rec_res:
                txts += [rec_res[i][0] for i in range(len(rec_res))]
                scores += [rec_res[i][1] for i in range(len(rec_res))]
        final_text = ' '.join(txts)  # Only assign here if there are cropped images

    print(f"Total detection time: {total_detection_time:.4f} seconds")
    print(f"Total recognition time: {total_recognition_time:.4f} seconds")
    print(f"Total CPU usage for detection: {total_detection_cpu:.4f} %")
    print(f"Total CPU usage for recognition: {total_recognition_cpu:.4f} %")
    
    return final_text

