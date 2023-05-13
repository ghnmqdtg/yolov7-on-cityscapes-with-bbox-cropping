#!/bin/bash

echo "Step 1: Download Cityscapes Dataset"
# Login to Cityscapes, you should provide your username and password
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username={your_username}&password={your_password}&submit=Login' https://www.cityscapes-dataset.com/login/
# Download `gtFine_trainvaltest.zip` with packageID 1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
# Download `leftImg8bit_trainvaltest.zip` with packageID 3
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3


echo "Step 2: Setup Cityscapes Dataset"
# Create folders that `cityscapes-to-coco-conversion` prefer
mkdir -p cityscapes-to-coco-conversion/data/cityscapes/annotations
# Move downloaded datasets to required path
mv gtFine cityscapes-to-coco-conversion/data/cityscapes
mv leftImg8bit cityscapes-to-coco-conversion/data/cityscapes
# Run the conversion
echo "Running the conversion to get bbox (it takes some time) ..."
python main.py --dataset cityscapes --datadir cityscapes-to-coco-conversion/data/cityscapes --outdir cityscapes-to-coco-conversion/data/cityscapes/annotations

echo "Step 3: Convert Cityscapes Dataset into YOLO format"
# Change back to the previous directory
# Create a new subfolder
mkdir -p yolov7/customdata/images
# Copy leftImg8bit to yolov7 folder
cp -r cityscapes-to-coco-conversion/data/cityscapes/leftImg8bit/* yolov7/customdata/images
cp cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_train.json yolov7/customdata/images/train/_annotations.coco.json
cp cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_val.json yolov7/customdata/images/val/_annotations.coco.json
# Run the format converter
echo "Running the format converter..."
python ./scripts/dataset_format_converter.py --source-folder-path ./yolov7/customdata/images/

echo "Process complete."
