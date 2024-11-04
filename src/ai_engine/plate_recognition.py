from typing import Dict, Any
from core.prj_config import prj_config
import numpy as np
import openvino as ov
from core.run_single_image import run_paddle_ocr_with_plate
from core.crop_plate import crop_image
import time
import os
import sys



#Modify part
def get_resource_path(relative_path):
    """ Get the absolute path to a resource in a PyInstaller bundle. """
    # Check if running in a PyInstaller bundle
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class PlateRecognition:

    def __init__(self,
                det_model_file_path: str = None,
                rec_model_file_path: str = None,
                ):
        self.det_model_file_path = det_model_file_path
        self.rec_model_file_path = rec_model_file_path
        
        self.load_model()
            
    
    def load_model(self):
      # self.plate_detector = Detector(model_path=self.detector_model_path, score_thr=prj_config.SCORE_THRESHOLD)
            # self.recognizer = Recognizer(ocr_path=self.ocr_model_path,
            #                              corner_path=self.corner_mode_path)
        core = ov.Core()
            # Load model detection

        det_model_file_path = get_resource_path("/home/hquanhh/data_plate/ai-engine-main/pretrained/det2/inference.pdmodel")
            # Initialize OpenVINO Runtime for text detection.
        det_model = core.read_model(model=det_model_file_path)
        det_compiled_model = core.compile_model(model=det_model, device_name="CPU")

            # Get input and output nodes for text detection.
        det_input_layer = det_compiled_model.input(0)
        det_output_layer = det_compiled_model.output(0)
            #
        rec_model_file_path = get_resource_path("/home/hquanhh/data_plate/ai-engine-main/pretrained/rec/inference.pdmodel")
            # Read the model and corresponding weights from a file.
        rec_model = core.read_model(model=rec_model_file_path)

            # Assign dynamic shapes to every input layer on the last dimension.
        for input_layer in rec_model.inputs:
            input_shape = input_layer.partial_shape
            input_shape[3] = -1
            rec_model.reshape({input_layer: input_shape})

        rec_compiled_model = core.compile_model(model=rec_model, device_name="CPU")

            # Get input and output nodes.
        rec_input_layer = rec_compiled_model.input(0)
        rec_output_layer = rec_compiled_model.output(0)
            

    def get_result(self, frame: np.ndarray) -> Dict[str, Any]:
        if not frame.shape:
            return None

        # Gọi hàm crop_image để lấy cropped_images và saved_boxes
        cropped_images, saved_boxes = crop_image(frame)

        # Gọi hàm run_paddle_ocr_with_plate với cả hai tham số
        results = run_paddle_ocr_with_plate(frame, saved_boxes, use_popup=False)

        if not results:
            return None

        return {
            "img_crop_org": "",
            "cls_id": "",
            "bbox": "",
            "score": "",
            "recognizer_result": results
        }






model_path_xml = get_resource_path('/home/hquanhh/data_plate/ai-engine-main/pretrained/plate/model_SDD.xml')
model_path_bin = get_resource_path('/home/hquanhh/data_plate/ai-engine-main/pretrained/plate/model_SDD.bin')