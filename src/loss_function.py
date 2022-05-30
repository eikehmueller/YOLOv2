"""YOLOv2 loss function

The loss function implemented here is based on the discussion in Yumi's blog
https://fairyonice.github.io/Part_4_Object_Detection_with_Yolo_using_VOC_2012_data_loss.html,
which is in turn follows the implementation by experiencor
https://github.com/experiencor/keras-yolo2/blob/master/Yolo%20Step-by-Step.ipynb.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras


class YOLOv2Loss(keras.losses.Loss):
    """Class for YOLOv2 loss function

    The loss function is constructed as the following sum over tiles (i,j) and
    anchor boxes k:

    loss = sum_{i,j,k} ( lambda_{coord} * loss^{(coord)}_{i,j,k}
                       + lambda_{obj} * loss^{(obj)}_{i,j,k}
                       + lambda_{noobj} * loss^{(noobj)}_{i,j,k}
                       + lambda_{classes} * loss^{(classes)}_{i,j,k}

    where

    loss^{(coord)}_{i,j,k} = ( (xc^{(pred)}_{i,j,k} - xc^{(true)}_{i,j,k})^2
                             + (yc^{(pred)}_{i,j,k} - yc^{(true)}_{i,j,k})^2
                             + (sqrt(w^{(pred)}_{i,j,k}) - sqrt(w^{(true)}_{i,j,k}))^2
                             + (sqrt(h^{(pred)}_{i,j,k}) - sqrt(h^{(true)}_{i,j,k}))^2 )
                           * C^{(true)}_{i,j,k} / N_{obj}
    loss^{(obj)}_{i,j,k} = ( C^{(true)}_{i,j,k} * IoU(bbox^{(true)}_{i,j,k},bbox^{(pred)}_{i,j.k})
                             - C^{(pred)}_{i,j,k} )^2 * C^{(true)}_{i,j,k} / N_{conf}
    loss^{(noobj)}_{i,j,k} = ( C^{(noobj)}_{i,j,k} )^2 * (1 - C^{(true)}_{i,j,k}) / N_{conf}
    loss^{(classes)}_{i,j,k} = - C^{(true)}_{i,j,k} / N_{obj}
                               * sum_{classes c} [ p^{(true)}_{i,j,k}(c)
                               * log ( p^{(true)}_{i,j,k}(c) ) ]

    Here N is the total number of tiles and N_{obj} = sum_{i,j,k} C^{(true)}_{i,j,k} is the
    total number of ground truth bounding boxes.
    C^{(noobj)}_{i,j,k} is constructed as follows:

    C^{(noobj)}_{i,j,k} = 1 if C^{(true)}_{i,j,k} = 0 AND
                    max_{i',j',k'} IoU(bbox^{(true)}_{i',j',k'},bbox^{(pred)}_{i,j.k}) < 0.6
    C^{(noonj)}_{i,j,k} = 0 otherwise.

    Further, N_{conf} = sum_{i,j,k} C^{(noobj)}_{i,j,k}.

    :arg anchor_boxes: coordinates of anchor boxes
    :arg lambda_coord: scaling factor for coordinate loss
    :arg lambda_obj: scaling factor for object loss
    :arg lambda_noobj: scaling factor for no-object loss
    :arg lambda_classes: scaling factor for classes loss
    :arg bbox_cachesize: maximal number of true bounding boxes that are stored in
                         compressed form to compute the IoU between all predicted and
                         ground truth bounding boxes in loss^{(noobj)}_{i,j,k}
    """

    def __init__(
        self,
        anchor_boxes,
        lambda_coord=1.0,
        lambda_obj=5.0,
        lambda_noobj=1.0,
        lambda_classes=1.0,
        bbox_cachesize=16,
    ):
        super().__init__()
        # Extract widths and heights of anchor boxes into numpy arrays
        n_anchors = len(anchor_boxes)
        self.anchor_wh = np.reshape(
            np.asarray(
                [
                    [anchor["width"] for anchor in anchor_boxes],
                    [anchor["height"] for anchor in anchor_boxes],
                ]
            ).T,
            (1, 1, 1, n_anchors, 2),
        )
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_classes = lambda_classes
        self.bbox_cachesize = bbox_cachesize

    def call(self, y_true, y_pred):
        """Evaluate loss function

        Returns the loss function described above, reducing over all dimensions (except
        possibly the batch dimension).

        Assuming the batch index is b, the true target contains the following values:

          * y_true(b,i,j,k,0:4): coordinates of bounding box associated with k-th
                                 anchor box in tile (i,j), encoded as decsribed below.
          * y_true(b,i,j,k,4):   1 if anchor box k in tile (i,j) contains an object, 0 otherwise
          * y_true(b,i,j,k,5+c): 1 if the object is of class c, 0 otherwise.

        the predicted target contains the following values:

          * y_pred(b,i,j,k,0:4): coordinates of bounding box associated with k-th
                                 anchor box in tile (i,j), encoded as decsribed below.
          * y_pred(b,i,j,k,4):   the prediction for the confidence
                                 P(object)*IoU(b_true,b_predicted)
          * y_pred(b,i,j,k,5+c): the logit of the probability of the object being of class c

        The true bounding box coordinates are encoded as follows:

          * xc^{(true)}_{i,j,k} = y_true(b,i,j,k,0)
            is the x-coordinate of the centre of the true bounding box, measured from the
            corner of tile (i,j) and in units of the tile size. Hence 0 <= xc_{i,j,k} <= 1.
          * yc^{(true)}_{i,j,k} = y_true(b,i,j,k,1)
            is the y-coordinate of the centre of the bounding box, measured from the corner of
            tile (i,j) and in units of the tile size. Hence 0 <= yc_{i,j,k} <= 1.
          * w^{(true)}_{i,j,k} = y_true(b,i,j,k,2)
            is the width of the true bounding box, measured in units of the tilde width.
            Hence 0 <= w_{i,j,k} <= n_tiles.
          * h^{(true)}_{i,j,k} = y_true(b,i,j,k,3)
            is the height of the true bounding box, measured in units of the tile height.
            Hence 0 <= h_{i,j,k} <= n_tiles.

        We also write p^{(true)}_{i,j,k}(c) = y_true(b,i,j,k,5+c) for the true probability and
        C^{(true)}_{i,j,k} = y_true(b,i,j,k,4) for the true confidence.

        Let further

          * t^{(x)}_{i,j,k} = y_pred(b,i,j,k,0)
          * t^{(y)}_{i,j,k} = y_pred(b,i,j,k,1)
          * t^{(w)}_{i,j,k} = y_pred(b,i,j,k,2)
          * t^{(h)}_{i,j,k} = y_pred(b,i,j,k,3)

        and denote the width and height of the k-th anchor box with w^{(a)}_k and h^{(a)}_k
        (both are measured in units of the tile width/height). Then the predicted bounding
        box coordinates are encoded like this:

          * xc^{(pred)}_{i,j,k} = sigmoid(t^{(x)}_{i,j,k})
          * yc^{(pred)}_{i,j,k} = sigmoid(t^{(y)}_{i,j,k})
          * w^{(pred)}_{i,j,k} = w^{(a)}_k * exp(t^{(w)}_{i,j,k})
          * h^{(pred)}_{i,j,k} = h^{(a)}_k * exp(t^{(h)}_{i,j,k})

        The predicted confidence is encoded as C^{(pred)}_{i,j,k} = sigmoid(y_pred(b,i,j,k,4)).
        The predicted probability of class c is given as

        p^{(pred)}_{i,j,k}(c) = N_{i,j,k}*exp(y_pred(b,i,j,k,5+c)) where N_{i,j,k} is a
        normalisation factor (this is automatically computed when converting to class
        probabilities with tensorflow's softmax function).

        :arg y_true: true target, shape (batchsize,n_tiles,n_tiles,n_anchor,5+n_classes)
        :arg y_pred: predicted target, shape (batchsize,n_tiles,n_tiles,n_anchor,5+n_classes)
        """
        # add batch dimension, if this is not already present
        if int(tf.rank(y_true)) < 5:
            y_pred = tf.expand_dims(y_pred, axis=0)
            y_true = tf.expand_dims(y_true, axis=0)
        # now the tensors y_true and y_pred should be of shape
        # (batchsize,n_tiles,n_tiles,n_anchors,5+n_classes)
        y_shape = y_true.get_shape().as_list()
        n_tiles = y_shape[1]
        # extract predicted bounding box width and height of bounding box (need to multiply by
        # anchor widths and heights and by number of tiles to convert to units
        # of tile width/height)
        bbox_wh = n_tiles * tf.exp(y_pred[..., 2:4]) * self.anchor_wh
        bbox_pred = {
            "xc": tf.sigmoid(y_pred[..., 0]),
            "yc": tf.sigmoid(y_pred[..., 1]),
            "width": bbox_wh[..., 0],
            "height": bbox_wh[..., 1],
        }
        # predicted confidence
        confidence_pred = tf.sigmoid(y_pred[..., 4])
        # predicted class probabilities
        classes_pred = tf.nn.softmax(y_pred[..., 5:], axis=-1)
        # true bounding box
        bbox_true = {
            "xc": y_true[..., 0],
            "yc": y_true[..., 1],
            "width": y_true[..., 2],
            "height": y_true[..., 3],
        }
        # true probability that anchor box contains object (=> is either 0 or 1)
        confidence_true = y_true[..., 4]
        # true class probabilities
        classes_true = y_true[..., 5:]
        # number of bounding boxes with objects
        N_obj = tf.reduce_sum(confidence_true, axis=(1, 2, 3), keepdims=True)
        # extract the bboxes of all ground truth objects

        # ==== 1. coordinate loss ====
        loss_coord = tf.math.divide_no_nan(confidence_true, N_obj) * (
            (bbox_pred["xc"] - bbox_true["xc"]) ** 2
            + (bbox_pred["yc"] - bbox_true["yc"]) ** 2
            + (tf.sqrt(bbox_pred["width"]) - tf.sqrt(bbox_true["width"])) ** 2
            + (tf.sqrt(bbox_pred["height"]) - tf.sqrt(bbox_true["height"])) ** 2
        )
        # ==== 2. no-object loss ====
        # Compute compressed tensor with true bounding boxes
        compressed_bbox_true = self.compress_bounding_boxes(confidence_true, bbox_true)
        threshold = 0.6
        # noobj(i,j,k) = 1 if { confidence_true(i,j,k) = 0 AND maxIoU(i,j,k) < threshold }
        # where maxIoU = max_{i',j',k'} IoU(bbox_true(i',j',k'),bbox_pred(i,j,k))
        bbox_pred_extended = {
            key: tf.expand_dims(bbox_pred[key], axis=-1) for key in bbox_pred.keys()
        }
        noobj = (1.0 - confidence_true) * (
            tf.cast(
                tf.reduce_max(
                    self._IoU(compressed_bbox_true, bbox_pred_extended),
                    axis=-1,
                )
                < threshold,
                confidence_true.dtype,
            )
        )
        N_noobj = tf.reduce_sum(noobj, axis=(1, 2, 3), keepdims=True)
        N_conf = N_obj + N_noobj
        loss_noobj = tf.math.divide_no_nan(noobj, N_conf) * confidence_pred**2
        # ==== 3. object loss ====
        iou = self._IoU(bbox_true, bbox_pred)
        loss_obj = (
            tf.math.divide_no_nan(confidence_true, N_conf)
            * (confidence_pred - iou * confidence_true) ** 2
        )
        # ==== 4. class cross-entropy loss ====
        loss_classes = -tf.math.divide_no_nan(confidence_true, N_obj) * tf.reduce_sum(
            classes_true * tf.math.log(classes_pred), axis=-1
        )
        # Construct total loss, summed over all dimensions (apart possibly from the
        # batch dimension)
        return tf.math.reduce_sum(
            self.lambda_coord * loss_coord
            + self.lambda_obj * loss_obj
            + self.lambda_noobj * loss_noobj
            + self.lambda_classes * loss_classes,
            axis=(1, 2, 3),
        )

    def compress_bounding_boxes(self, confidence, bbox):
        """Construct tensor of shape (batchsize,1,1,1,batch_size*bbox_cachesize) to encode
        __all__ true bounding boxes in an image.

        This uses tf.gather_nd() and tf.scatter_nd() to scatter all values corresponding to
        true bounding boxes into tensors of shape (batchsize,batch_size*bbox_cachesize), which
        are then reshaped to (batchsize,1,1,1,batch_size*bbox_cachesize).

        :arg confidence: confidence as a tensor of shape (batchsize,n_tiles,n_tiles,n_anchors)
        :arg bbox: dictionary with entries 'xc', 'yc', 'width', 'height', where each entry is a
                   tensor of shape (batchsize,n_tiles,1,1,1,n_tiles,n_anchors)
        """
        # work out the batch size
        batchsize = confidence.get_shape().as_list()[0]
        # work out the indices of all non-zero entries in the confidence tensor. This will
        # return an array of size (nnz,4) where nnz is the number of true bounding boxes
        indices = tf.where(confidence > 0).numpy()
        batch_indices = indices[:, 0]  # extract the batch indices
        nnz = len(batch_indices)  # number of nonzero bounding boxes
        # next, construct the indices in the compressed array by assigning a consecutive list
        # of indices to each batch index
        compressed_indices = np.zeros((nnz, 2), dtype=np.int64)
        for j in range(batchsize):
            n_batch_indices = np.sum(batch_indices == j)
            batch_j = (
                batch_indices == j
            )  # the set of indices belonging to a particular image
            compressed_indices[:, 0][batch_j] = j
            compressed_indices[:, 1][batch_j] = range(n_batch_indices)
        # construct dictionary with compressed entries using gather_nd() and scatter_nd()
        return {
            key: tf.reshape(
                tf.scatter_nd(
                    indices=compressed_indices,
                    updates=tf.gather_nd(bbox[key], indices=indices),
                    shape=(batchsize, batchsize * self.bbox_cachesize),
                ),
                (batchsize, 1, 1, 1, batchsize * self.bbox_cachesize),
            )
            for key in bbox.keys()
        }

    @tf.function
    def _IoU(self, bbox_1, bbox_2):
        """Helper function to compute IoU of two bounding boxes.

        Note that the method works for tensors of any shape, i.e. the values in the
        bbox_1 and bbox_2 dictionaries can be tensor-valued.

        If the corners of bounding box j are given by

        (x^{(min)}_j, y^{(min)}_j) and (x^{(max)}_j, y^{(max)}_j)

        then the area of intersection is:

        A_{int} = max ( min(x^{(max)}_1,x^{(max)}_1) - max(x^{(min)}_1,x^{(min)}_2), 0)
                * max ( min(y^{(max)}_1,y^{(max)}_1) - max(y^{(min)}_1,y^{(min)}_2), 0)

        and the areas of the j-th bounding box is

        A_j = ( x^{(max)}_j - x^{(min)}_j ) * ( y^{(max)}_j - y^{(min)}_j )

        This method returns the intersection over union, given by

        IoU = A_{int} / (A_1 + A_2 - A_{int})

        :arg bbox_1: First bounding box, given as a dictionary
                     {'xc':centre_x,'yc':centre_y,'width':width,'height':height}
        :arg bbox_2: Second bounding box, given as a dictionary in the same format
        """
        # Convert to coordinates of four corners
        xmin_1 = bbox_1["xc"] - 0.5 * bbox_1["width"]
        ymin_1 = bbox_1["yc"] - 0.5 * bbox_1["height"]
        xmax_1 = bbox_1["xc"] + 0.5 * bbox_1["width"]
        ymax_1 = bbox_1["yc"] + 0.5 * bbox_1["height"]
        xmin_2 = bbox_2["xc"] - 0.5 * bbox_2["width"]
        ymin_2 = bbox_2["yc"] - 0.5 * bbox_2["height"]
        xmax_2 = bbox_2["xc"] + 0.5 * bbox_2["width"]
        ymax_2 = bbox_2["yc"] + 0.5 * bbox_2["height"]
        # Compute area of intersection
        area_intersection = tf.math.maximum(
            (tf.math.minimum(xmax_1, xmax_2) - tf.math.maximum(xmin_1, xmin_2)),
            0.0,
        ) * tf.math.maximum(
            (tf.math.minimum(ymax_1, ymax_2) - tf.math.maximum(ymin_1, ymin_2)),
            0.0,
        )
        # compute areas of individual bounding boxes
        area_1 = tf.multiply(bbox_1["width"], bbox_1["height"])
        area_2 = tf.multiply(bbox_2["width"], bbox_2["height"])
        return area_intersection / (area_1 + area_2 - area_intersection)
