#!/bin/bash

echo "Step 1: Download Cityscapes Dataset"
echo "1-1: Login to Cityscapes, you should provide your username and password"
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username={your_username}&password={your_password}&submit=Login' https://www.cityscapes-dataset.com/login/
echo "1-2: Download `gtFine_trainvaltest.zip` with packageID 1"
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
echo "1-3: Download `leftImg8bit_trainvaltest.zip` with packageID 3"
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
echo "1-4: Unzip the downloaded files"
unzip gtFine_trainvaltest.zip
unzip leftImg8bit_trainvaltest.zip

echo "Step 2: Setup Cityscapes Dataset"
echo "2-1: Create folders that `cityscapes-to-coco-conversion` prefer"
mkdir -p cityscapes-to-coco-conversion/data/cityscapes/annotations
echo "2-2: Move downloaded datasets to required path"
mv gtFine cityscapes-to-coco-conversion/data/cityscapes
mv leftImg8bit cityscapes-to-coco-conversion/data/cityscapes
echo "Running the conversion to get COCO annotations and bbox (it takes some time) ..."
python main.py --dataset cityscapes --datadir cityscapes-to-coco-conversion/data/cityscapes --outdir cityscapes-to-coco-conversion/data/cityscapes/annotations

echo "Step 3: Convert Cityscapes Dataset into YOLO format"
# Change back to the previous directory
echo "3-1: Create a new subfolder"
mkdir -p yolov7/customdata/images
echo "3-2: Copy the images and annotations to the yolov7 subfolder"
cp -r cityscapes-to-coco-conversion/data/cityscapes/leftImg8bit/* yolov7/customdata/images
cp cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_train.json yolov7/customdata/images/train/_annotations.coco.json
cp cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_val.json yolov7/customdata/images/val/_annotations.coco.json
# Run the format converter
echo "3-3: Running the COCO to YOLO format converter..."
python ./scripts/dataset_format_converter.py --source-folder-path ./yolov7/customdata/images/

echo "Process complete."
