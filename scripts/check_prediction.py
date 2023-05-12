import json
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import torch
import sys
pred_path = sys.argv[1]
target_path = sys.argv[2]
def get_gt_data(imageID, annotations):
    gt_data = {'boxes':[],
               'labels':[],
               'image_id':[],
               'area':[],
               'iscrowd':[]}
    
    for annotation in annotations:
        if annotation["image_id"] == imageID:
            bbox = annotation['bbox']
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            gt_data['boxes'].append(bbox)
            gt_data['labels'].append(annotation['category_id'])
            gt_data['image_id'].append(imageID)
            gt_data['area'].append(annotation['area'])
            gt_data['iscrowd'].append(annotation['iscrowd'])
    return gt_data

with open(pred_path, 'r') as f:
    preds = json.load(f)

with open(target_path, 'r') as f:
    gt = json.load(f)
fname_to_imageID = {}

for image in gt["images"]:
    fname_to_imageID[image["file_name"]] = image["id"]
metric = MeanAveragePrecision()
device = 'cuda'
for fname, pred in tqdm(preds.items()):
    pred = [{k: torch.tensor(v).to(device) for k, v in pred.items()}]
    imageID = fname_to_imageID[fname]
    target = get_gt_data(imageID, gt["annotations"])
    target = [{k: torch.tensor(v).to(device) for k, v in target.items()}]
    metric.update(pred, target)
    print(pred)
    print(target)
result = metric.compute()
print(result)
