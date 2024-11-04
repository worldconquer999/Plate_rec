import itertools

import cv2
import numpy as np

from openvino.inference_engine import IECore

from src.core.prj_config import BASE_DIR
from src.core.license_utils import load_content_decode


default_model_path = BASE_DIR + r"/pretrained/ocr.model"
default_weights_path = BASE_DIR + r"/pretrained/ocr.weights"


class OCRDecoder:
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
        self.init_map_char()

    def init_map_char(self):
        letters = " 1234567890QWERTYIUPLKJHGFDASZXCVBNM-."
        index = 0
        char2index = {}

        for i in letters:
            char2index[i] = index
            index += 1
        index2char = {}

        for u, v in char2index.items():
            index2char[v] = u

        self.char2index = char2index
        self.index2char = index2char

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

    def labels_to_text(self, labels):
        re = ""
        for i in labels:
            try:
                m = self.index2char[i]
                re += m
            except:
                continue
        return re

    def decode_batch(self, out):
        ret = []
        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = self.labels_to_text(out_best)
            ret.append(outstr)
        return ret

    @staticmethod
    def process_data_before_infer(x):
        """
            NOTE: not processed in batch
        """
        # image = np.array(x)
        image = cv2.resize(x, (128, 32))

        image = np.expand_dims(image, -1)/255.
        image = np.array(image, dtype=np.float32, order='C')

        image = image.reshape((32, 128))

        return image

    def predict(self, batch_x):
        """
            batch_x: shape (batch_size, w, h) // Gray image
            type numpy
        """
        image = self.process_data_before_infer(batch_x)

        res = self.exec_net.infer(inputs={self.input_blob: image})

        output_shapes = (1, 32, 39)

        res = res[list(res.keys())[0]]
        res = res.reshape(output_shapes)
        res = np.mean(res, axis=0, keepdims=True)

        return res
