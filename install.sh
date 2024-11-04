#!/bin/bash

# Install common python package
pip3 install --upgrade wheel pip setuptools &&
pip3 install -r requirements.txt &&

# Install YOLOX object detection
cd src/ai_engine/YOLOX &&
pip3 install -v -e .