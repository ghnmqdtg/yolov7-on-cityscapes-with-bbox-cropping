#!/bin/bash

echo "Create Conda Env"
conda create --name yolov7_with_cropping python=3.10
echo "Activate Conda Env"
conda activate yolov7_with_cropping
echo "Install torch and CUDA"
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# The VSCdoe interpreter path is ~/.conda/envs/yolov7_with_cropping/bin/python

echo "Install the required packages"
pip install -r yolov7/requirements.txt
pip install -r cityscapes-to-coco-conversion/requirements.txt
pip install wandb
pip install pycocotools
pip install torchmetrics