import numpy as np
from openvino.inference_engine import IECore

import cv2
from src.core.prj_config import BASE_DIR
from src.core.license_utils import load_content_decode

default_model_path = BASE_DIR + r"/pretrained/corner.model"
default_weights_path = BASE_DIR + r"/pretrained/corner.weights"

class CornerDetector:
    def __init__(self, model_path, weights_path=None):
        if model_path is None:
            self.model_path = default_model_path
        else:
            self.model_path = model_path

        if weights_path is None:
            self.weights_path = default_weights_path
        else:
            self.weights_path = weights_path

        self.load_model()

    def load_model(self):
        ie = IECore()

        model = load_content_decode(self.model_path)
        weights = load_content_decode(self.weights_path)

        self.net = ie.read_network(model=model, weights=weights, init_from_buffer=True)
        self.input_blob = next(iter(self.net.input_info))
        self.out_blob = next(iter(self.net.outputs))

        self.net.input_info[self.input_blob].precision = 'FP32'
        self.net.outputs[self.out_blob].precision = 'FP16'

        self.exec_net = ie.load_network(network=self.net, device_name="CPU")

    @staticmethod
    def process_data_before_infer(x):
        """
            NOTE: not processed in batch
        """
        # image = np.array(x)
        image = cv2.resize(x, (256, 256))

        image = np.expand_dims(np.array(x), -1) / 255.
        image = image.reshape((256, 256))

        image = np.array(image, dtype=np.float32, order='C')

        return image

    def predict(self, batch):
        """
            batch_x: shape (batch_size, w, h) // Gray image
            type numpy
        """
        image = self.process_data_before_infer(batch)

        res = self.exec_net.infer(inputs={self.input_blob: image})

        output_shapes = (1, 64, 64, 2)

        res = np.array([res[key] for key in list(res.keys())])
        res = res.reshape(output_shapes)
        res = np.argmax(res, axis=-1)

        return res
