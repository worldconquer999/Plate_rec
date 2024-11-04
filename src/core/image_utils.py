import base64
import pickle
import numpy as np
import cv2
import uuid
import os
import time

from core.model import InferRequestImgType
from core.prj_config import prj_config

def base64_to_numpy(img_base64):
    img_converted = base64.b64decode(img_base64)
    img_converted = np.frombuffer(img_converted, dtype=np.uint8)
    img = cv2.imdecode(img_converted, flags=1)
    return img


def save_image(img):
    filename = f"{str(uuid.uuid4())}-{int(time.time())}.jpg"
    path = os.path.join(prj_config.SAVE_PATH, filename)
    cv2.imwrite(path, img)


def get_numpy_frame(binary_frame, img_type):
    if img_type == InferRequestImgType.PICKLE:
        numpy_frame = pickle.loads(binary_frame)
    elif img_type == InferRequestImgType.BASE64:
        numpy_frame = base64_to_numpy(binary_frame)
    elif img_type == InferRequestImgType.BYTE:
        numpy_frame = np.frombuffer(binary_frame, dtype=np.uint8)
        numpy_frame = cv2.imdecode(numpy_frame, cv2.IMREAD_COLOR)
    else:
        raise Exception(
            "Unsupported img type",
        )
    if not numpy_frame.size:
        raise Exception(
            "No frame",
        )
    return numpy_frame
