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
One major difference is the implementation of the computation of the term in the loss function which computes the IoU of a predicted bounding box $\mathcal{B} (x_{i,j},y_{i,j},\w_{i,j},h_{i,j})$ with all ground truth bounding boxes and checks that the maximum IoU does not exceed 0.6. Instead of using the trick in [Yumi's blog](https://fairyonice.github.io/Part_4_Object_Detection_with_Yolo_using_VOC_2012_data_loss.html), namely passing a tensor with all true bounding boxes through the network, this tensor is constructed when evaluating the loss function. For this, the indices of all true bounding boxes $\mathcal{B}(x_{i',j'},y_{i',j'},w_{i',j'},h_{i',j'})$ are extracted from the ground truth tensor, which has shape $M\times S \times S \times B\times(5+C)$ andused to construct a tensor of shape $M\times M\cdot N_{\text{buffer}}$ with `tf.gather_nd()` and `tf.scatter_nd()` calls. Here $M=8$ is the batchsize (I couldn't train with larger batches due to memory limitations) $S=13$ is the number of gridboxes in each direction, $B=5$ is the number of anchor boxes and $C$ is the number of classes ($C=80$ for COCO and $C=20$ for PascalVOC). $N_{\text{buffer}}$ is the size of the true anchor box buffer, and ideally this should be larger than the total number of true anchor boxes.

### Weights
Joseph Redmon's [weights for Darknet](https://pjreddie.com/darknet/yolo/) can be downloaded with
```
!wget https://pjreddie.com/media/files/yolov2.weights
```

### Anchor box generation
The anchor boxes are generated with k-means clustering, as described in the paper. This is done for both the PascalVOC and for the COCO datasets in `GenerateAnchorBoxes.ipynb`. The results are stored in the json files `anchor_boxes_coco.json` and `anchor_boxes_pascalvoc.json`.

## Datasets
The code can process images from the [COCO](https://cocodataset.org/#home) and [PascalVOC](https://cocodataset.org/#home) datasets. I used the 2017 train/val data from COCO and the VOC2012 data. The relevant classes `COCOImageReader` and `PascalVOCImageReader` are implemented in `image_reader.py`

## Code structure

## References
* Redmon, J., Divvala, S., Girshick, R. and Farhadi, A., 2016. *You only look once: Unified, real-time object detection.* In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788). [arxiv:1506.02640](https://arxiv.org/abs/1506.02640)
* Redmon, J. and Farhadi, A., 2017. *YOLO9000: better, faster, stronger.* In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7263-7271).[arxiv:1612.08242](https://arxiv.org/abs/1612.08242)

## Loss function
The loss function is central for a training and it took me a while to understand it properly since it is not given explicitly in the original YOLOv2 paper. The following discussion of the loss function is based on [link to Yumi's blog](https://fairyonice.github.io/Part_4_Object_Detection_with_Yolo_using_VOC_2012_data_loss.html), which is in turn based on the [implementation by experiencor](https://github.com/experiencor/keras-yolo2/blob/master/Yolo%20Step-by-Step.ipynb), fixing some typos along the way.

### Total loss function
The total loss is given by

$$
\text{loss} = \sum_{i=1}^{S^2}\sum_{j=1}^B \left(\text{loss}_{i,j}^{xywh} + \text{loss}_{i,j}^p + \text{loss}_{i,j}^c\right)
$$


Here $i=1,\dots,S^2$ is the index of the gridcell and $j=1,\dots,B$ is the index of the anchor box slot. Each of the three terms in the loss function will be scaled by a hyperparameter; these hyperparameters are denoted as $\lambda_{\text{coord}}$, $\lambda_{\text{class}}$ and $\lambda_{\text{obj}}$

Let $C_{i,j}$ be the ground truth that there is an object associated with anchor box $j$ in grid cell $i$. The the total number of objects in the image is given by

$$
N_{\text{obj}} = \sum_{i=1}^{S^2}\sum_{j=1}^B C_{i,j}.
$$

### Loss due to bounding box mismatch (coordinate loss)
$$
\text{loss}_{i,j}^{xywh} = \frac{\lambda_{\text{coord}}}{N_{\text{obj}}} C_{i,j}\left[\left(x_{i,j}-\hat{x}_{i,j}\right)^2+\left(y_{i,j}-\hat{y}_{i,j}\right)^2+\left(\sqrt{w_{i,j}}-\sqrt{\hat{w}_{i,j}}\right)^2+\left(\sqrt{h_{i,j}}-\sqrt{\hat{h}_{i,j}}\right)^2\right]
$$

Here $x_{i,j}$, $y_{i,j}$, $w_{i,j}$, $h_{i,j}$ are the true coordinates of the centre and width/height of the bounding box. The corresponding predicted values $\hat{x}_{i,j}$, $\hat{y}_{i,j}$, $\hat{w}_{i,j}$, $\hat{h}_{i,j}$ are indicated with a hat. Since each term is multiplied by $C_{i,j}\in\{0,1\}$, the coordinate loss only contributes for those $i,j$ which correspond to a true bounding box.

### Classification loss

Let $p_{i,j}^{c}\in\{0,1\}$ be the ground truth probability that the object associated with $i,j$ is of class $c\in\text{classes}$. The corresponding predicted probabilities $\hat{p}_{i,j}^c$ are denoted with a hat. Then the classification loss in $i,j$ is given by the cross-entropy

$$
\text{loss}_{i,j}^{c} = -\frac{\lambda_{\text{class}}}{N_{\text{obj}}} C_{i,j} \sum_{c\in\text{classes}} p_{i,j}^c \log\left(\hat{p}_{i,j}^c\right).
$$

Again, since we multiply each term by $C_{i,j}\in\{0,1\}$, only those $i,j$ which are associated with a real object contribute.

### Confidence loss

Define

$$
C_{i,j}^{\text{noobj}} = \begin{cases}
1 & \text{if $\max_{i',j'}\left\{\text{IoU}\left(\mathcal{B}(x_{i',j'},y_{i',j'},w_{i',j'},h_{i',j'}),\mathcal{B}(\hat{x}_{i,j},\hat{y}_{i,j},\hat{w}_{i,j},\hat{h}_{i,j})\right)\right\} < 0.6$ and $C_{i,j}=0$}\\
0 & \text{otherwise}
\end{cases}.
$$

Here $\mathcal{B}(x,y,w,h)$ denotes the bounding box with centre coordinate $x,y$ and width/height $w,h$. $\text{IoU}\left(\mathcal{B}_a,\mathcal{B}_b\right)$ is the *''intersection over union''* of two bounding boxes $\mathcal{B}_a$ and $\mathcal{B}_b$.

Further, let

$$
N^{\text{conf}} = \sum_{i=1}^{S^2}\sum_{j=1}^B\left(C_{i,j}+C_{i,j}^{\text{noobj}}\right).
$$

Then the confidence loss is

$$
\begin{aligned}
\text{loss}_{i,j}^{c} &= \frac{\lambda_{\text{obj}}}{N^{\text{conf}}} C_{i,j}\left(\text{IoU}\left(\mathcal{B}(x_{i,j},y_{i,j},w_{i,j},h_{i,j}),\mathcal{B}(\hat{x}_{i,j},\hat{y}_{i,j},\hat{w}_{i,j},\hat{h}_{i,j})\right)-\hat{C}_{i,j}\right)^2\\
&\quad+\;\; \frac{\lambda_{\text{noobj}}}{N^{\text{conf}}} C_{i,j}\left(0-\hat{C}_{i,j}\right)^2
\end{aligned}
$$

where $\hat{C}_{i,j}$ is the predicted confidence of finding an object in $i,j$. Note that I think that in [Yumi's blog](https://fairyonice.github.io/Part_4_Object_Detection_with_Yolo_using_VOC_2012_data_loss.html) a square is missing in the second line.

### Final comments
I believe that there is a bug in Yumi's `get_conf_mask()` function: the penultimate line should read 
```Python
conf_mask = conf_mask + true_box_conf * LAMBDA_OBJECT
```
(i.e. `true_box_conf_IOU` needs to be replaced by `true_box_conf`) to be consistent with the experiencor's implementation and the discussion in the blog.

## todo
* write code structure section
