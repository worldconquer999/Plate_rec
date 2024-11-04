from core.plate_and_text_detection import infer_and_visualize
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

def crop_image(image_path):
    cropped_images = []  # List to store cropped images
    image, saved_boxes = infer_and_visualize(model_path_xml, model_path_bin, image_path)  # Ensure this returns the image and boxes
    for box_info in saved_boxes:
        start_point, end_point = box_info['box']
        
        # Extract x_min, y_min, x_max, y_max from start_point and end_point
        x_min, y_min = start_point
        x_max, y_max = end_point

        # Crop the image using OpenCV
        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_images.append(cropped_image)  # Save cropped image into the list
    
    return cropped_images, saved_boxes
