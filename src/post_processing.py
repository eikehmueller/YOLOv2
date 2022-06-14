"""Post processing tools such non-max-suppression"""
from itertools import compress
import numpy as np


def iou(bbox_1, bbox_2):
    """Compute IoU of two different bounding boxes

    :arg bbox_1: first bounding box, characterised by corner (xc,yc) an width/height
    :arg bbox_2: second bounding box
    """
    xmin_1 = bbox_1["xc"] - 0.5 * bbox_1["width"]
    ymin_1 = bbox_1["yc"] - 0.5 * bbox_1["height"]
    xmax_1 = bbox_1["xc"] + 0.5 * bbox_1["width"]
    ymax_1 = bbox_1["yc"] + 0.5 * bbox_1["height"]
    xmin_2 = bbox_2["xc"] - 0.5 * bbox_2["width"]
    ymin_2 = bbox_2["yc"] - 0.5 * bbox_2["height"]
    xmax_2 = bbox_2["xc"] + 0.5 * bbox_2["width"]
    ymax_2 = bbox_2["yc"] + 0.5 * bbox_2["height"]
    # Compute area of intersection
    area_intersection = max((max(xmax_1, xmax_2) - max(xmin_1, xmin_2)), 0.0) * max(
        (max(ymax_1, ymax_2) - max(ymin_1, ymin_2)), 0.0
    )
    # compute areas of individual bounding boxes
    area_1 = bbox_1["width"] * bbox_1["height"]
    area_2 = bbox_2["width"] * bbox_2["height"]
    return area_intersection / (area_1 + area_2 - area_intersection)


def non_max_suppression(bboxes, overlap_threshold=0.5, confidence_threshold=0.5):
    """Run non-max suppression on list of bounding boxes

    The bounding boxes are given as a list of dictionaries, with each dictionary
    enconding the bounding coordinates (keys "xc" and "yc" for corner of bounding box,
    keys "width" and "height" for width and height), confidence (key "confidence") and
    class index (key "class").

    Non-max suppression is applied to return a list of updated bounding boxes in the same
    format.

    :arg bboxes: list of bounding boxes for a single image
    :arg overlap_threshold: threshold on IoU to consider overlap
    :arg confidence_threshold: discard all bounding boxes with a confidence lower than
         this threshold
    """
    # Bounding boxes after non-max suppression
    nms_bboxes = []
    # Work out the set of classes that are present in the image
    bbox_classes = {bbox["class"] for bbox in bboxes}
    # Loop over bounding boxes for each class
    for bboxes_per_class in [
        [bbox for bbox in bboxes if bbox["class"] == cls] for cls in bbox_classes
    ]:
        # Sort bounding boxes for each class in order of decreasing confidence
        bboxes_per_class.sort(key=lambda x: x["confidence"], reverse=True)
        while len(bboxes_per_class) > 0:
            first_bbox = bboxes_per_class.pop(0)
            if first_bbox["confidence"] < confidence_threshold:
                break
            nms_bboxes.append(first_bbox)
            # only keep bounding boxes that have an IoU with the most confident prediction
            # that is lower than the overlap_threshold
            bboxes_per_class = list(
                compress(
                    bboxes_per_class,
                    [
                        iou(bbox, first_bbox) < overlap_threshold
                        for bbox in bboxes_per_class
                    ],
                )
            )
    return nms_bboxes


def soft_non_max_suppression(bboxes, sigma=0.5, confidence_threshold=0.01):
    """Run soft non-max suppression on list of bounding boxes

    Implements the soft non-max-suppression with Gaussian kernel described in
    N. Bodla, B. Singh, R. Chellappa, L. S. Davis (2017) "Soft-NMS -- Improving Object
    Detection With One Line of Code" https://arxiv.org/abs/1704.04503

    The bounding boxes are given as a list of dictionaries, with each dictionary
    enconding the bounding coordinates (keys "xc" and "yc" for corner of bounding box,
    keys "width" and "height" for width and height), confidence (key "confidence") and
    class index (key "class").

    Soft non-max suppression is applied to return a list of updated bounding boxes in the same
    format.

    :arg bboxes: list of bounding boxes for a single image
    :arg sigma: width of Gaussian used to decay confidences
    :arg confidence_threshold: discard all bounding boxes with a confidence lower than
         this threshold
    """
    # Bounding boxes after non-max suppression
    nms_bboxes = []
    # Work out the set of classes that are present in the image
    bbox_classes = {bbox["class"] for bbox in bboxes}
    # Loop over bounding boxes for each class
    for bboxes_per_class in [
        [bbox for bbox in bboxes if bbox["class"] == cls] for cls in bbox_classes
    ]:
        while len(bboxes_per_class) > 0:
            # Sort bounding boxes for each class in order of decreasing confidence
            bboxes_per_class.sort(key=lambda x: x["confidence"], reverse=True)
            first_bbox = bboxes_per_class.pop(0)
            if first_bbox["confidence"] < confidence_threshold:
                break
            nms_bboxes.append(first_bbox)
            # Decay confidences of remaining bounding boxes
            for bbox in bboxes_per_class:
                decay_factor = np.exp(-0.5 * iou(bbox, first_bbox) ** 2 / sigma**2)
                bbox.update(confidence=decay_factor * bbox["confidence"])
    return nms_bboxes
