#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Megvii, Inc. and its affiliates.

import cv2
import numpy as np

from openvino.inference_engine import IECore

import sys
from src.core.prj_config import BASE_DIR, prj_config
from src.core.license_utils import load_content_decode
sys.path.append(BASE_DIR + r'/src/ai_engine/YOLOX')

from yolox.data.data_augment import preproc as preprocess
from yolox.utils import multiclass_nms, demo_postprocess

default_model_path = BASE_DIR + r"/pretrained/yolox.model"
default_weights_path = BASE_DIR + r"/pretrained/yolox.weights"


class Detector:
    def __init__(self, model_path: str = None, weights_path: str = None, score_thr=0.5):
        if model_path is None:
            self.model_path = default_model_path
        else:
            self.model_path = model_path

        if weights_path is None:
            self.weights_path = default_weights_path
        else:
            self.weights_path = weights_path

        self.score_thr = score_thr
        self.load_model()

    def load_model(self):
        ie = IECore()

        model = load_content_decode(self.model_path)
        weights = load_content_decode(self.weights_path)

        self.net = ie.read_network(model=model, weights=weights, init_from_buffer=True)
        self.input_blob = next(iter(self.net.input_info))
        self.out_blob = next(iter(self.net.outputs))

        # Set input and output precision manually
        self.net.input_info[self.input_blob].precision = 'FP32'
        self.net.outputs[self.out_blob].precision = 'FP16'

        # Get a number of classes recognized by a model
        self.num_of_classes = max(self.net.outputs[self.out_blob].shape)
        self.exec_net = ie.load_network(network=self.net, device_name="CPU")

        _, _, self.h, self.w = self.net.input_info[self.input_blob].input_data.shape

    def process_result_yolox(self, res, ratio):

        predictions = demo_postprocess(res, (self.h, self.w), p6=False)[0]

        boxes_ = predictions[:, :4]
        scores = predictions[:, 4, None] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes_)
        boxes_xyxy[:, 0] = boxes_[:, 0] - boxes_[:, 2]/2.
        boxes_xyxy[:, 1] = boxes_[:, 1] - boxes_[:, 3]/2.
        boxes_xyxy[:, 2] = boxes_[:, 0] + boxes_[:, 2]/2.
        boxes_xyxy[:, 3] = boxes_[:, 1] + boxes_[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)

        return dets

    @staticmethod
    def get_crop_bboxs(origin_img, boxes, cls_ids, scores, conf=0.5):
        crop_frames = []

        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            x0 = max(x0 - prj_config.PADDING, 0)
            y0 = max(y0 - prj_config.PADDING, 0)
            x1 = x1 + prj_config.PADDING
            y1 = y1 + prj_config.PADDING

            img_crop = origin_img[y0:y1, x0:x1]
            crop_frames.append((img_crop, cls_id, (x0, y0, x1, y1), score))

        return crop_frames

    def detect(self, origin_img):
        image, ratio = preprocess(origin_img, (self.h, self.w))
        res = self.exec_net.infer(inputs={self.input_blob: image})
        res = res[self.out_blob]

        dets = self.process_result_yolox(res, ratio)

        if dets is not None:
            final_boxes = dets[:, :4]
            final_scores, final_cls_inds = dets[:, 4], dets[:, 5]
            crop_frames = self.get_crop_bboxs(
                origin_img, final_boxes, final_cls_inds, final_scores, self.score_thr)
            return crop_frames
        else:
            return []
