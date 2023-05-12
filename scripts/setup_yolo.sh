echo "Load miniconda3"
module load miniconda3
echo "Create Conda Env"
conda create --name yolo_with_cropping python=3.10
echo "Activate Conda Env"
conda activate yolo_with_cropping
echo "Install torch and CUDA"
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# The VSCdoe interpreter path is ~/.conda/envs/yolo_with_cropping/bin/python

# echo "Clone the YOLOv7"
git clone https://github.com/WongKinYiu/yolov7.git
echo "Install the required packages"
pip install -r yolov7/requirements.txt
pip install rebox
pip install pycocotools
pip install torchmetrics