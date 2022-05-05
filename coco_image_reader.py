import json
from pycocotools.coco import COCO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from anchor_boxes import BestAnchorBoxFinder


class COCOImageReader(object):
    """Class for reading and resizing images with bbox annotations from COCO dataset

    This class allows reading images from the COCO-dataset and generating bespoke
    annotations for object detection. These annotations can then be converted to tensors
    which can be used as targets for YOLOv2
    """

    def __init__(
        self,
        data_dir="../../cocodata/",
        data_type="val2017",
        image_size=416,
        n_tiles=13,
        verbose=False,
    ):
        """Create new instance

        :arg data_dir: directory containing the data. This is assumed to contain the
                       subdirectories /annotations and /images
        :arg data_type: type of COCO data to read, for example "val2017" for the 2017
                       validation data
        :arg image_size: size of resized images
        :arg n_tiles: number of tiles each dimension is subdivided into
        :arg verbose: print additional information
        """
        self.image_size = image_size
        self.n_tiles = n_tiles
        annotations_file = f"{data_dir}/annotations/instances_{data_type}.json"
        self.image_dir = f"{data_dir}/images/{data_type}/"
        self.coco = COCO(annotations_file)

        categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_name_map = {x["id"]: x["name"] for x in categories}
        self.category_idx_map = {x["id"]: j for j, x in enumerate(categories)}
        self.n_classes = len(self.category_idx_map)
        print(f"number of classes = {self.n_classes}")
        self.all_category_names = set([category["name"] for category in categories])
        if verbose:
            print("COCO categories: \n{}\n".format(", ".join(self.all_category_names)))
        with open("anchor_boxes.json", "r", encoding="utf8") as f:
            self.anchors = json.load(f)
        self.n_anchor = len(self.anchors)
        self.babf = BestAnchorBoxFinder(self.anchors)

    def get_image_ids(self, category_names=None):
        """Return the ids of all images in the dataset which belong to a certain category

        :arg category_names: names of categories that should appear in the images
        """
        # Check that categories are ok, i.e. they are a subset of all categories in the COCO dataset
        assert (category_names is None) or set(category_names).issubset(
            self.all_category_names
        ), "invalid category name(s)"
        if category_names is None:
            category_names = ["any"]
        category_ids = self.coco.getCatIds(catNms=category_names)
        img_ids = self.coco.getImgIds(catIds=category_ids)
        return img_ids

    def read_image(self, img_id):
        """Read an image, resize it and return the resized image together with bbox annotations

        Returns a dictionary of the form {'image':(CV2 image object),'bboxes'(bbox annotations)}

        The bounding box annotations consist of a list of dictionaries of the following form:
        { 'category_id':   the id of the labelled object category according to COCO
          'category_name': the name of the category for the labelled object
          'class':         the class index, which is a number in the range 0,1,...,n_cat-1
                           where n_cat is the number of categories in the dataset
          'xc':            the x-coordinate of the center of the bbox
          'yc':            the y-coordinate of the center of the bbox
          'width':         the width of the bounding box
          'height':        the height of the bounding box
        }

        :arg img_id: id of image to process
        """
        img = self.coco.loadImgs([img_id])[0]
        image = cv2.resize(
            cv2.imread(self.image_dir + "/" + img["file_name"]),
            (self.image_size, self.image_size),
        )
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        annotations = self.coco.loadAnns(ann_ids)
        bboxes = []
        for annotation in annotations:
            label = self.category_name_map[annotation["category_id"]]
            idx = self.category_idx_map[annotation["category_id"]]
            xmin, ymin, width, height = annotation["bbox"]
            bboxes.append(
                {
                    "category_id": annotation["category_id"],
                    "class": idx,
                    "category_name": label,
                    "xc": round(self.image_size / img["width"] * (xmin + 0.5 * width)),
                    "yc": round(
                        self.image_size / img["height"] * (ymin + 0.5 * height)
                    ),
                    "width": round(self.image_size / img["width"] * width),
                    "height": round(self.image_size / img["height"] * height),
                }
            )
        return {"image": image, "bboxes": bboxes}

    def create_target(self, bboxes):
        """Create a target for list of bounding boxes (obtained with read_image)

        The target y is an array of shape (n_tiles,n_tiles,n_anchor,5+n_classes)
        where
          * n_tiles is the number of tiles the image is subdivided into
          * n_anchor is the number of anchor boxes
          * n_classes is the number of classes

        A bounding box is associated with a tile if the centre of the bounding box lies in that tile

        Let i=(i_x,i_y) be the index of the tile and j be the anchor box. Then we have that:

        xc and yc are measured relative to the corner of the tile that contains the center
        of the object and they are normalised to the tile size, i.e. xc, yc in [0,1].

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
                y_target[tile_x, tile_y, j, 2] = bboxes[k]["width"] / self.image_size
                y_target[tile_x, tile_y, j, 3] = bboxes[k]["height"] / self.image_size
                y_target[tile_x, tile_y, j, 4] = 1
                y_target[tile_x, tile_y, j, 5 + bboxes[k]["class"]] = 1
        return y_target

    def show_annotated_image(self, annotated_image):
        """Auxilliary function for displaying an annotated image generated by the read_image()
        method

        :arg annotated_image: annotated image as returned by read_image()
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(annotated_image["image"])
        ax = plt.gca()
        ax.set_xlim(0, self.image_size)
        ax.set_ylim(self.image_size, 0)
        for bbox in annotated_image["bboxes"]:
            ax.add_patch(
                mpl.patches.Rectangle(
                    (
                        bbox["xc"] - 0.5 * bbox["width"],
                        bbox["yc"] - 0.5 * bbox["height"],
                    ),
                    bbox["width"],
                    bbox["height"],
                    facecolor="none",
                    edgecolor="yellow",
                )
            )
            plt.plot(bbox["xc"], bbox["yc"], marker="x", markersize=8, color="yellow")
            plt.text(
                bbox["xc"] - 0.5 * bbox["width"],
                bbox["yc"] - 0.5 * bbox["height"],
                bbox["category_name"],
                color="yellow",
            )
        ax.set_xticks(self.image_size / self.n_tiles * np.arange(self.n_tiles))
        ax.set_xticklabels(range(self.n_tiles))
        ax.set_yticks(self.image_size / self.n_tiles * np.arange(self.n_tiles))
        ax.set_yticklabels(range(self.n_tiles))
        plt.grid()
        plt.show()
