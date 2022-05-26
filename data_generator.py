import random
import numpy as np
import tensorflow as tf
from anchor_boxes import BestAnchorBoxFinder


def inv_sigmoid(y):
    """Inverse of sigmoid function

    Auxilliary function to compute x, given y = sigma(x) = 1/(1+exp(-x))

    :arg y: Value of sigma function
    """
    return np.log(y) - np.log(1.0 - y)


class DataGeneratorFactory(object):
    """Class for generating data in the tensorflow dataset format

    Provides generator for pairs (X,y) of images and bounding box annotations. Here X is
    a RGB image of shape (image_size,image_size,3) and y is a tensor of shape
    (n_tiles,n_tiles,n_anchor,5+n_classes) such that:

        * y(b,i,j,k,0:4): coordinates of bounding box associated with k-th
                          anchor box in tile (i,j), encoded as decsribed below.
        * y(b,i,j,k,4):   1 if anchor box k in tile (i,j) contains an object, 0 otherwise
        * y(b,i,j,k,5+c): 1 if the object is of class c, 0 otherwise.

    (see documentation of the bbox2targets() method below for more details).

    The resulting dataset can then be used to loop over all images and corresponding
    bounding box annotations (converted to targets) in the dataset.
    """

    def __init__(self, anchors, image_reader, random_shuffle=False, max_images=None):
        """Create new instance

        The image_reader class is assumed to implement the following two methods:
          * get_image_ids() to get the ids of all images
          * read_image() to return an image together with its bounding box annotations

        :arg anchors: list of anchor boxes
        :arg image_reader: image reader instance.
        :arg random_shuffle: randomly shuffle the image ids when iterating over the dataset
        :arg max_images: limit the number of images that we iterate over (for testing)
        """
        # image reader
        self.image_reader = image_reader
        self.image_ids = self.image_reader.get_image_ids()

        # extract size information
        self.image_size = self.image_reader.image_size
        self.n_tiles = self.image_reader.n_tiles
        self.n_classes = self.image_reader.n_classes

        # set anchors and initialise the best anchor box finder, which is used
        # to identify the optimal anchor box for each bounding box in an image
        self.anchors = anchors
        self.n_anchor = len(self.anchors)
        self.babf = BestAnchorBoxFinder(self.anchors)
        # Extract widths and heights of anchor boxes into numpy arrays (the is required
        # for scaling the data)
        self.anchor_wh = np.asarray(
            [
                [anchor["width"] for anchor in self.anchors],
                [anchor["height"] for anchor in self.anchors],
            ]
        ).T
        self.random_shuffle = random_shuffle
        if max_images:
            self.max_images = max_images
        else:
            self.max_images = self.image_reader.get_n_images()
        # tensorflow dataset
        self.dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_signature=(
                tf.TensorSpec(
                    shape=(self.image_size, self.image_size, 3),
                    dtype=tf.float32,
                ),
                tf.TensorSpec(
                    shape=(
                        self.n_tiles,
                        self.n_tiles,
                        self.n_anchor,
                        5 + self.n_classes,
                    ),
                    dtype=tf.float32,
                ),
            ),
        )

    def _generator(self):
        """Generate a new sample (X,y)"""
        ids = list(self.image_ids)
        while True:
            if self.random_shuffle:
                random.shuffle(ids)
            for image_id in ids[: self.max_images]:
                annotated_image = self.image_reader.read_image(image_id)
                yield annotated_image["image"], self.bboxes2target(
                    annotated_image["bboxes"]
                )

    def bboxes2target(self, bboxes):
        """Create a target for list of bounding boxes

        The list of bounding boxes is generated with the read_image() method of the
        COCOImageReader class.

        The target y is an array of shape (n_tiles,n_tiles,n_anchor,5+n_classes)
        where
          * n_tiles is the number of tiles the image is subdivided into
          * n_anchor is the number of anchor boxes
          * n_classes is the number of classes

        A bounding box is associated with a tile if the centre of the bounding box lies in that tile

        Let i=(i_x,i_y) be the index of the tile and j be the anchor box. Then we have that:

        xc and yc are measured relative to the corner of the tile that contains the center
        of the object and they are normalised to the tile size, i.e. xc, yc are in [0,1].
        The width and height are measured in units of the tile size also.

          xc_{i,j} := y(i_x,i_y,j,0) is the x- coordinate of the center of the
                                       bounding box paired with the j-th anchor
          yc_{i,j} := y(i_x,i_y,j,1) is the x- coordinate of the center
          w_{i,j} := y(i_x,i_y,j,2) is the width of the bounding box
          h_{i,j} := y(i_x,i_y,j,3) is the height of the bounding box
          C_{i,j} := y(i_x,i_y,j,4) = 1 if the j-the anchor box contains an object and
                                        zero otherwise
          p_{i,j,k} := y(i_x,i_y,j,5+k) = 1 if the object in the box has class k and
                                            zero otherwise

        :arg bbox: bounding box annotations, in the format generated by the read_image()
                   method
        """
        # Size of tiles in the image
        tile_size = self.image_size // self.n_tiles
        tiled_annotations = {}
        for bbox in bboxes:
            # Work out the tile that contains the bbox centre
            tile_x = bbox["xc"] // tile_size
            tile_y = bbox["yc"] // tile_size
            tile_id = (tile_x, tile_y)
            if tile_id in tiled_annotations.keys():
                tiled_annotations[tile_id].append(bbox)
            else:
                tiled_annotations[tile_id] = [
                    bbox,
                ]
        # Now populate the target tensor y
        y_target = np.zeros(
            (self.n_tiles, self.n_tiles, self.n_anchor, 5 + self.n_classes)
        )
        for tile_id, bboxes in tiled_annotations.items():
            tile_x, tile_y = tile_id
            # Match bounding box to the anchor with the best overlap
            # j = anchor box index
            # k = bounding box index
            anchor_ind, bbox_ind = self.babf.match(bboxes)
            for j, k in zip(anchor_ind, bbox_ind):
                xc = bboxes[k]["xc"] / tile_size - tile_x
                yc = bboxes[k]["yc"] / tile_size - tile_y
                y_target[tile_x, tile_y, j, 0] = xc
                y_target[tile_x, tile_y, j, 1] = yc
                y_target[tile_x, tile_y, j, 2] = (
                    bboxes[k]["width"] / self.image_size * self.n_tiles
                )
                y_target[tile_x, tile_y, j, 3] = (
                    bboxes[k]["height"] / self.image_size * self.n_tiles
                )
                y_target[tile_x, tile_y, j, 4] = 1
                y_target[tile_x, tile_y, j, 5 + bboxes[k]["class"]] = 1
        return y_target

    def prediction2bboxes(self, y_pred, threshold=0.5):
        """Convert prediction generated by the neural network to list of bounding boxes

        y_pred is a four dimensional array which stores the bounding box coordinates,
        confidence scores and class probabilities as follows. First introduce the variables

          * t^{(x)}_{i,j,k} = y_pred(i,j,k,0)
          * t^{(y)}_{i,j,k} = y_pred(i,j,k,1)
          * t^{(w)}_{i,j,k} = y_pred(i,j,k,2)
          * t^{(h)}_{i,j,k} = y_pred(i,j,k,3)

        and denote the width and height of the k-th anchor box with w^{(a)}_k and h^{(a)}_k
        (both are measured in units of the tile width/height). Then the bounding
        box coordinates are encoded like this:

          * xc^{(pred)}_{i,j,k} = (sigmoid(t^{(x)}_{i,j,k}) + i) * tile_size
          * yc^{(pred)}_{i,j,k} = (sigmoid(t^{(y)}_{i,j,k}) + j) * tile_size
          * w^{(pred)}_{i,j,k} = w^{(a)}_k * exp(t^{(w)}_{i,j,k})
          * h^{(pred)}_{i,j,k} = h^{(a)}_k * exp(t^{(h)}_{i,j,k})

        The predicted confidence is encoded as C^{(pred)}_{i,j,k} = sigmoid(y_pred(i,j,k,4)).
        The predicted probability of class c is given as

        p^{(pred)}_{i,j,k}(c) = N_{i,j,k}*exp(y_pred(i,j,k,5+c)) where N_{i,j,k} is a
        normalisation factor (this is automatically computed when converting to class
        probabilities with tensorflow's softmax function).

        :arg y_pred: output of neural network, four dimensional tensor of shape
                     (n_tiles,n_tiles,n_anchor,5+n_classes)
        :arg threshold: discard any predictions with a confidence below this threshold
        """
        # Size of tiles in the image
        tile_size = self.image_size // self.n_tiles
        # width and height of bounding boxes
        bbox_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(
            self.anchor_wh,
            (len(tf.shape(y_pred)) - 2) * (1,) + (y_pred.shape[-2], 2),
        )
        # xc and ys are stored relative to corner of tile, so need to add offsets when
        # converting back to global coordinates
        offsets = np.asarray(range(self.n_tiles))
        offsets_x = np.reshape(offsets, (self.n_tiles, 1, 1))
        offsets_y = np.reshape(offsets, (1, self.n_tiles, 1))
        xc_pred = (tf.sigmoid(y_pred[..., 0]) + offsets_x) * tile_size
        yc_pred = (tf.sigmoid(y_pred[..., 1]) + offsets_y) * tile_size
        width_pred = self.image_size * bbox_wh[..., 0]
        height_pred = self.image_size * bbox_wh[..., 1]
        class_pred = tf.math.argmax(tf.nn.softmax(y_pred[..., 5:], axis=-1), axis=-1)
        # predicted confidence
        confidence_pred = np.asarray(tf.sigmoid(y_pred[..., 4]))
        # Only extract bounding boxes which have a confidence score that exceeds threshold
        bboxes = [
            {
                "class": class_pred[i, j, k].numpy(),
                "xc": xc_pred[i, j, k].numpy(),
                "yc": yc_pred[i, j, k].numpy(),
                "width": width_pred[i, j, k].numpy(),
                "height": height_pred[i, j, k].numpy(),
            }
            for i, j, k in np.asarray(np.where(confidence_pred > threshold)).transpose()
        ]
        return bboxes

    def bboxes2prediction(self, bboxes, noise=0.0):
        """Inverse of method prediction2bboxes

        Takes a set of bounding boxes and returns a prediction, i.e. a corresponding tensor
        of shape (n_tiles,n_tiles,n_anchor,5+n_classes). This prediction can be noisy in
        the sense that random noise is added to the tensor after the conversion.

        All bounding box coordinates are given in absolute coordinates, i.e. they are not
        scaled by image width or height.

        This method is mainly used for debugging.

        :arg bboxes: List of bounding box dictionaries of the form
                     {"class":class of object in box,
                      "xc":centre in x-direction,
                      "yc":centre in y-direction,
                      "width":width,
                      "height":height}
        :arg noise: width or normal distribution of random noise that is added to the
                    prediction tensor
        """
        tile_size = self.image_size // self.n_tiles
        tiled_annotations = {}
        for bbox in bboxes:
            # Work out the tile that contains the bbox centre
            tile_x = bbox["xc"] // tile_size
            tile_y = bbox["yc"] // tile_size
            tile_id = (tile_x, tile_y)
            if tile_id in tiled_annotations.keys():
                tiled_annotations[tile_id].append(bbox)
            else:
                tiled_annotations[tile_id] = [
                    bbox,
                ]
        # Now populate the prediction tensor y
        y_pred = np.zeros(
            (self.n_tiles, self.n_tiles, self.n_anchor, 5 + self.n_classes)
        )
        # set logits of no-objects to -100
        y_pred[..., 4] = -100.0
        for tile_id, bboxes in tiled_annotations.items():
            tile_x, tile_y = tile_id
            # Match bounding box to the anchor with the best overlap
            # j = anchor box index
            # k = bounding box index
            anchor_ind, bbox_ind = self.babf.match(bboxes)
            for j, k in zip(anchor_ind, bbox_ind):
                xc = bboxes[k]["xc"] / tile_size - tile_x
                yc = bboxes[k]["yc"] / tile_size - tile_y
                y_pred[tile_x, tile_y, j, 0] = inv_sigmoid(xc)
                y_pred[tile_x, tile_y, j, 1] = inv_sigmoid(yc)
                y_pred[tile_x, tile_y, j, 2] = np.log(
                    bboxes[k]["width"] / (self.anchors[j]["width"] * self.image_size)
                )
                y_pred[tile_x, tile_y, j, 3] = np.log(
                    bboxes[k]["height"] / (self.anchors[j]["height"] * self.image_size)
                )
                # arbitrarily set the logit of the prediction to 100
                y_pred[tile_x, tile_y, j, 4] = 100.0
                # arbitrarily set the logit of the true class to 10.
                # (which means that the true class is 22000x more likely than any
                # other class)
                y_pred[tile_x, tile_y, j, 5 + bboxes[k]["class"]] = 10.0
        y_pred[...] += np.random.normal(
            scale=noise,
            size=(self.n_tiles, self.n_tiles, self.n_anchor, 5 + self.n_classes),
        )
        return y_pred
