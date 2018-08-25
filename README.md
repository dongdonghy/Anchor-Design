## Design anchors with biggest recall and IOU for training set labels

### Overview
The scales and ratios of anchor are important during anchor-based detector. This script generates some metric like recall rate and IOU for a set of anchors and groundtruth labels. 

### Dependencies
* torch
* numpy
* xml

### Anchor Metrics
* recall rate
* average IOU of max IOUs between gt and anchors
* average positive anchors
* average IOU of positive anchors

### How to Run
* put training labels in labels directory
* change the anchor ratio and scale to what you want
* run:
``` python
python anchor_design.py
```

