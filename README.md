# YOLOv7 on Cityscapes with bbox cropping
## Introduction
In this project, I trained YOLOv7 with the Cityscapes dataset and implemented the bbox cropping feature. It's an experimental idea to convert the segmentation annotations to bbox annotations with `cityscapes-to-coco-conversion` and then convert the COCO annotations to yolo annotations with my scripts and train the model with bbox annotations for object detection. The bbox cropping feature is implemented by modifying the `detect.py` file.

The model is trained with 100 epochs, and the mAP@50 is 0.61266, which is lower than using the COCO dataset. I think the reason is that the Cityscapes dataset has many small objects since the camera is mounted on the car, and there always are objects far from the point of view.

## Environment
- Python 3.10.11
- Pytorch 1.13.1
- Torchvision 0.14.1
- CUDA 11.7

## Setup

1. Clone the project and its submodules
    
    ```bash
    $ git clone --recurse-submodules https://github.com/ghnmqdtg/yolov7-on-cityscapes-with-bbox-cropping.git
    ```
    
2. Go into the project folder
    
    ```bash
    $ cd yolov7-on-cityscapes-with-bbox-cropping
    ```

3. Run `./scripts/setup_env.sh` to setup the env.
    
    ```bash
    $ sh scripts/setup_env.sh
    ```

    - Create a conda env named `yolov7_with_cropping` with python 3.10.11.
    
    - Install pytorch with cuda 11.7.
    
    - Install the dependencies.
    
4. (Optional) Change VSCode interpreter path with `~/.conda/envs/yolov7_with_cropping/bin/python`.
    
5. Modify the `./scripts/setup_dataset.sh` line 5 with your cityscapes username and password.
    
6. Run `./scripts/setup_dataset.sh` to setup the env; this takes some time.
    
    ```bash
    $ sh scripts/setup_dataset.sh
    ```

    - Download the dataset.
    
    - Use `cityscapes-to-coco-conversion` to generate  bbox annotations of Cityscapes dataset using segmentation annotations. (Cityscapes has no bbox annotations).

    - Convert annotations from COCO format to YOLO format.

7. Download the pretrained model and put it to `./yolov7` folder.
    
    ```bash
    $ wget https://github.com/ghnmqdtg/yolov7-on-cityscapes-with-bbox-cropping/releases/download/v0.1/yolov7_cityscapes.pt \
        -O ./yolov7/yolov7_cityscapes.pt
    ```


## Train and evaluate the model
1. You should `cd` to `yolov7` folder first
    
    ```bash
    $ cd yolov7
    ```

2. Train the model with cityscapes
    
    ```bash
    $ python -m torch.distributed.launch \
        --nproc_per_node 1 \
        --master_port 9527 \
        train.py \
        --workers 2 \
        --device 0 \
        --sync-bn \
        --epochs 100 \
        --batch-size 32 \
        --data data/cityscape.yaml \
        --img 640 640 \
        --cfg cfg/training/yolov7.yaml \
        --weights ./yolov7.pt \
        --hyp data/hyp.scratch.p5.yaml
    ```

    The output will be in `runs/train`.

3. Evaluation
    
    ```bash
    $ python test.py \
        --data data/cityscape.yaml \
        --img 640 \
        --batch 32 \
        --conf 0.001 \
        --iou 0.65 \
        --device 0 \
        --weights yolov7_cityscapes.pt \
        --name cityscapes_yolo_cityscapes
    ```
    
    The output will be in `runs/test`.

## Run inference

- On single image
    
    ```bash
    $ python detect.py \
        --weights yolov7_cityscapes.pt \
        --conf 0.25 \
        --img-size 640 \
        --source customdata/images/test/bonn/bonn_000004_000019_leftImg8bit.png
    ```
    
    The output will be in `runs/detect`.

- On a video

    Nope, I haven't tried it yet.

## Experimental Results

<table align="center" width="100%" border="0">
    <tr>
        <td colspan="2" style="text-align:center; font-size:14px;"><b>Training & Evaluation Report<b></td>
    </tr>
    <tr>
        <td width="50%" style="text-align:center;font-size:14px;"><b>mAP@50: 0.61266<b></td>
        <td width="50%" style="text-align:center;font-size:14px;"><b>mAP@50:95 : 0.38005)<b></td>
    </tr>
    <tr>
        <td><img src="imgs/yolo_cityscapes_map50.png"></img></td>
        <td><img src="imgs/yolo_cityscapes_map50_95.png"></img></td>
    </tr>
    <tr>
        <td colspan="3" width="33%" style="text-align:center;font-size:14px;"><b>Confusion Matrix<b></td>
    </tr>
    <tr>
        <td colspan="3"><img src="imgs/confusion_matrix.png"></img></td>
    </tr>
    <tr>
        <td width="50%" style="text-align:center;font-size:14px;"><b>F1 curve<b></td>
        <td width="50%" style="text-align:center;font-size:14px;"><b>PR curve<b></td>
    </tr>
    <tr>
        <td><img src="imgs/F1_curve.png"></img></td>
        <td><img src="imgs/PR_curve.png"></img></td>
    </tr>
    <tr>
        <td width="50%" style="text-align:center;font-size:14px;"><b>P curve<b></td>
        <td width="50%" style="text-align:center;font-size:14px;"><b>R curve<b></td>
    </tr>
    <tr>
        <td><img src="imgs/P_curve.png"></img></td>
        <td><img src="imgs/R_curve.png"></img></td>
    </tr>
  </table>
