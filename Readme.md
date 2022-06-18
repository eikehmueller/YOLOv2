[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# tensorflow implementation of YOLO v2
This repository contains an implementation of YOLOv2 and is based on [Vivek Maskara's blog](https://www.maskaravivek.com/post/yolov2/), which in turn follow [Yumi's blog](https://fairyonice.github.io/Part_4_Object_Detection_with_Yolo_using_VOC_2012_data_loss.html). The backbone network is [Joseph Redmon's Darknet](https://pjreddie.com/darknet/), but in this repository it is of course implemented in tensorflow.

### Prerequisites
The following Python modules are required to run the code
* [tensorflow 2](https://www.tensorflow.org/)
* [OpenCV for Python](https://pypi.org/project/opencv-python/)
* [COCO API for Python](https://github.com/cocodataset/cocoapi)

### Loss function evaluation
Mathematical details on the loss function can be found in [this notebook](LossFunction.ipynb).

One major difference is the implementation of the computation of the term in the confidence loss
which computes the IoU of a predicted bounding box $\mathcal{B} (x^p_{i,j},y^p_{i,j},w^p_{i,j},h^p_{i,j})$ with all ground truth bounding boxes and checks that the maximum IoU does not exceed 0.6. Instead of using the trick in [Yumi's blog](https://fairyonice.github.io/Part_4_Object_Detection_with_Yolo_using_VOC_2012_data_loss.html), namely passing a tensor with all true bounding boxes through the network, this tensor is constructed when evaluating the loss function. For this, the indices of all true bounding boxes $\mathcal{B}(x_{i',j'},y_{i',j'},w_{i',j'},h_{i',j'})$ are extracted from the ground truth tensor, which has shape $M\times S \times S \times B\times(5+C)$, and used to construct a tensor of shape $M\times M\times N_{\text{buffer}}$ with `tf.gather_nd()` and `tf.scatter_nd()` calls. Here $M=8$ is the batchsize (I couldn't train with larger batches due to memory limitations) $S=13$ is the number of gridboxes in each direction, $B=5$ is the number of anchor boxes and $C$ is the number of classes ($C=80$ for COCO and $C=20$ for PascalVOC). $N_{\text{buffer}}$ is the size of the true anchor box buffer, and ideally this should be larger than the total number of true anchor boxes.



### Weights
Joseph Redmon's [weights for Darknet](https://pjreddie.com/darknet/yolo/) can be downloaded with
```
!wget https://pjreddie.com/media/files/yolov2.weights
```

### Anchor box generation
The anchor boxes are generated with k-means clustering, as described in the paper. This is done for both the PascalVOC and for the COCO datasets in `GenerateAnchorBoxes.ipynb`. The results are stored in the json files `anchor_boxes_coco.json` and `anchor_boxes_pascalvoc.json`.

## Datasets
The code can process images from the [COCO](https://cocodataset.org/#home) and [PascalVOC](https://cocodataset.org/#home) datasets. I used the 2017 train/val data from COCO and the VOC2012 data. The relevant classes `COCOImageReader` and `PascalVOCImageReader` are implemented in `image_reader.py`

## Example detection
![Sample detection](detection.png)

## Code structure

## References
* Redmon, J., Divvala, S., Girshick, R. and Farhadi, A., 2016. *You only look once: Unified, real-time object detection.* In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788). [arxiv:1506.02640](https://arxiv.org/abs/1506.02640)
* Redmon, J. and Farhadi, A., 2017. *YOLO9000: better, faster, stronger.* In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7263-7271).[arxiv:1612.08242](https://arxiv.org/abs/1612.08242)

## todo
* write code structure section
