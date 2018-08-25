import pdb
import os
import json
from lxml.etree import Element, SubElement, tostring, ElementTree
from xml.dom.minidom import parseString
import numpy as np
import torch
from generate_anchors import generate_anchors
from bbox_transform import bbox_overlaps_batch

label_dir='labels'
label_file = os.listdir(label_dir)                                                                                         
print('number of images:  ' + str(len(label_file)))

num_fg=0
total_iou=0
gt_over07=0
total_over07=0
total_iou07=0.0

# generate all anchors
anchor_scales=np.array([3,6,9,12,15,18,21,24,27,30,33])
ratios=np.array([2,3,4])
_anchors = torch.from_numpy(generate_anchors(scales=anchor_scales, ratios=ratios)).float()
feat_height, feat_width = 128, 256
shift_x = np.arange(0, feat_width) * 8
shift_y = np.arange(0, feat_height) * 8
shift_x, shift_y = np.meshgrid(shift_x, shift_y)
shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                          shift_x.ravel(), shift_y.ravel())).transpose())
shifts = shifts.contiguous().float()
A = 33
K = shifts.size(0)
all_anchors = _anchors.view(1, A, 4) + shifts.view(K, 1, 4)
all_anchors = all_anchors.view(K * A, 4)
all_anchors = all_anchors.float()

for i in range(len(label_file)):
    boxes=[]
    box=[]
    json_file= os.path.join(label_dir, label_file[i])
    with open(json_file, 'r') as f:
        data=json.load(f)
    for j in range(len(data['objects'])):
        if (float(data['objects'][j]['bboxVis'][2])*float(data['objects'][j]['bboxVis'][3]))/(float(data['objects'][j]['bbox'][2])*float(data['objects'][j]['bbox'][3]))>0.65 and float(data['objects'][j]['bbox'][3])>50 and data['objects'][j]['label']=='pedestrian':
            box=[float(data['objects'][j]['bbox'][0]), float(data['objects'][j]['bbox'][1]), float(data['objects'][j]['bbox'][0])+float(data['objects'][j]['bbox'][2])-1, float(data['objects'][j]['bbox'][1])+float(data['objects'][j]['bbox'][3])-1, 1.0]
            boxes.append(box)
    if len(boxes)>0:
        boxes_tensor=torch.zeros(1,len(boxes),5)
        boxes_tensor[0,:,:]=torch.from_numpy(np.array(boxes)).float()
        box_anchor_iou=bbox_overlaps_batch(all_anchors, boxes_tensor.view(1,len(boxes),5))
        gt_max_overlaps, _ = torch.max(box_anchor_iou, 1)
        print(gt_max_overlaps)
        total_iou+=gt_max_overlaps.sum()
        num_fg+=gt_max_overlaps.shape[1]
        gt_over07+=int(torch.sum(gt_max_overlaps>0.7))
        total_over07+=int(torch.sum(box_anchor_iou>0.7))
        total_iou07+=float(torch.sum(box_anchor_iou[box_anchor_iou>0.7]))

print('number of positive labels:  ' + str(num_fg))
print('average IOU of max IOU per gt:  ' + str(total_iou/num_fg))
print('recall:  ' + str(float(gt_over07)/num_fg))
print('average number of positive anchors:  ' + str(float(total_over07)/num_fg))
if total_over07>0:
    print('average IOU over 0.7:  ' + str(total_iou07/total_over07))
else:
    print('no over 07')
