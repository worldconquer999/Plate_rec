import os
import pickle
from cryptography.fernet import Fernet
from src.core.prj_config import BASE_DIR, prj_config


encrypt_folder = BASE_DIR + r"/pretrained"
if not os.path.exists(encrypt_folder):
    os.makedirs(encrypt_folder)


fernet = Fernet(prj_config.TOKEN)
paths = [
    [BASE_DIR + r"/pretrained_models/ocr96/namhoai96.xml", BASE_DIR + r"/pretrained/ocr96.model"],
    [BASE_DIR + r"/pretrained_models/ocr96/namhoai96.bin", BASE_DIR + r"/pretrained/ocr96.weights"],    
    [BASE_DIR + r"/pretrained_models/text_plate_det/best.xml", BASE_DIR + r"/pretrained/text_plate_det.model"],
    [BASE_DIR + r"/pretrained_models/text_plate_det/best.bin", BASE_DIR + r"/pretrained/text_plate_det.weights"],    
]

for path in paths:
    source, destination = path
    with open(source, "rb") as f:
        content = f.read()
    encMessage = fernet.encrypt(content)
    with open(destination, "wb") as f:
        pickle.dump(encMessage, f)