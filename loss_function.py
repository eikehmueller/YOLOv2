"""YOLOv2 loss function"""
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
    loss^{(obj)}_{i,j,k} = ( C^{(true)}_{i,j,k} * IoU(bbox^{(true)},bbox^{(true)})
                             - C^{(pred)}_{i,j,k} )^2 * C^{(true)}_{i,j,k} / N
    loss^{(noobj)}_{i,j,k} = ( C^{(pred)}_{i,j,k} )^2 * (1 - C^{(true)}_{i,j,k}) / N
    loss^{(classes)}_{i,j,k} = - C^{(true)}_{i,j,k} / N_{obj}
                               * sum_{classes c} [ p^{(true)}_{i,j,k}(c)
                               * log ( p^{(true)}_{i,j,k}(c) ) ]

    Here N is the total number of tiles and N_{obj} = sum_{i,j,k} C^{(true)}_{i,j,k} is the
    total number of ground truth bounding boxes.

    :arg anchor_boxes: coordinates of anchor boxes
    :arg lambda_coord: scaling factor for coordinate loss
    :arg lambda_obj: scaling factor for object loss
    :arg lambda_noobj: scaling factor for no-object loss
    :arg lambda_classes: scaling factor for classes loss
    """

    def __init__(
        self,
        anchor_boxes,
        lambda_coord=1.0,
        lambda_obj=5.0,
        lambda_noobj=1.0,
        lambda_classes=1.0,
    ):
        super().__init__()
        # Extract widths and heights of anchor boxes into numpy arrays
        self.anchor_wh = np.asarray(
            [
                [anchor["width"] for anchor in anchor_boxes],
                [anchor["height"] for anchor in anchor_boxes],
            ]
        )
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_classes = lambda_classes

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
            is the width of the true bounding box, measured in units of the total image width.
            Hence 0 <= w_{i,j,k} <= 1.
          * h^{(true)}_{i,j,k} = y_true(b,i,j,k,3)
            is the height of the true bounding box, measured in units of the total image height.
            Hence 0 <= h_{i,j,k} <= 1.

        We also write p^{(true)}_{i,j,k}(c) = y_true(b,i,j,k,5+c) for the true probability and
        C^{(true)}_{i,j,k} = y_true(b,i,j,k,4) for the true confidence.

        Let further

          * t^{(x)}_{i,j,k} = y_pred(b,i,j,k,0)
          * t^{(y)}_{i,j,k} = y_pred(b,i,j,k,1)
          * t^{(w)}_{i,j,k} = y_pred(b,i,j,k,2)
          * t^{(h)}_{i,j,k} = y_pred(b,i,j,k,3)

        and denote the width and height of the k-th anchor box with w^{(a)}_k and h^{(a)}_k
        (both are measured in units of the total image width/height). Then the predicted bounding
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
        y_shape = y_true.get_shape().as_list()
        # number of total boxes N_total = n_tiles * n_tiles * n_anchors
        N_total = np.prod(y_shape[-4:-1])
        # predicted bounding box
        # width and height of bounding box (need to multiply by anchor widths and heights)
        bbox_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(
            self.anchor_wh,
            (len(y_shape) - 2) * (1,) + (y_shape[-2], 2),
        )
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
        N_obj = tf.reduce_sum(confidence_true, axis=(-3, -2, -1), keepdims=True)

        # ==== 1. coordinate loss ====
        loss_coord = tf.math.divide_no_nan(confidence_true, N_obj) * (
            (bbox_pred["xc"] - bbox_true["xc"]) ** 2
            + (bbox_pred["yc"] - bbox_true["yc"]) ** 2
            + (tf.sqrt(bbox_pred["width"]) - tf.sqrt(bbox_true["width"])) ** 2
            + (tf.sqrt(bbox_pred["height"]) - tf.sqrt(bbox_true["height"])) ** 2
        )
        # ==== 2. object loss ====
        iou = self._IoU(bbox_true, bbox_pred)
        loss_obj = (
            confidence_true * (confidence_pred - iou * confidence_true) ** 2 / N_total
        )
        # ==== 3. no-object loss ====
        loss_noobj = (1.0 - confidence_true) * confidence_pred**2 / N_total
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
            axis=[-1, -2, -3],
        )

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
