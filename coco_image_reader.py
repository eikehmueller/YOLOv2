from pycocotools.coco import COCO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl


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
        # Map from category ids to classes (=class indices)
        self.category_class_map = {x["id"]: j for j, x in enumerate(categories)}
        # Map from classes (=class indices) to category names
        self.class_category_name_map = {j: x["name"] for j, x in enumerate(categories)}
        self.n_classes = len(self.category_class_map)
        print(f"number of classes = {self.n_classes}")
        self.all_category_names = set([category["name"] for category in categories])
        if verbose:
            print("COCO categories: \n{}\n".format(", ".join(self.all_category_names)))

    def get_image_ids(self, category_names=None):
        """Return the ids of all images in the dataset which belong to a certain category

        :arg category_names: names of categories that should appear in the images.
                             If None, use any category.
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

    def get_n_images(self, category_names=None):
        """Get the number of images which belong to a certain category

        :arg category_names: names of categories that should appear in the images.
                             If None, use any category.
        """
        return len(self.get_image_ids(category_names))

    def read_image(self, img_id):
        """Read an image, resize it and return the resized image together with bbox annotations

        Returns a dictionary of the form {'image':(CV2 image object),'bboxes'(bbox annotations)}

        The bounding box annotations consist of a list of dictionaries of the following form:
        { 'xc':            the x-coordinate of the center of the bbox
          'yc':            the y-coordinate of the center of the bbox
          'width':         the width of the bounding box
          'height':        the height of the bounding box
          'class':         the class index, which is a number in the range 0,1,...,n_cat-1
                           where n_cat is the number of categories in the dataset
        }

        All bounding box coordinates are given in absolute values (i.e. not scaled by image size)

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
            class_id = self.category_class_map[annotation["category_id"]]
            xmin, ymin, width, height = annotation["bbox"]
            bboxes.append(
                {
                    "class": class_id,
                    "xc": round(self.image_size / img["width"] * (xmin + 0.5 * width)),
                    "yc": round(
                        self.image_size / img["height"] * (ymin + 0.5 * height)
                    ),
                    "width": round(self.image_size / img["width"] * width),
                    "height": round(self.image_size / img["height"] * height),
                }
            )
        return {"image": image, "bboxes": bboxes}

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
                self.class_category_name_map[bbox["class"]],
                color="yellow",
            )
        ax.set_xticks(self.image_size / self.n_tiles * np.arange(self.n_tiles))
        ax.set_xticklabels(range(self.n_tiles))
        ax.set_yticks(self.image_size / self.n_tiles * np.arange(self.n_tiles))
        ax.set_yticklabels(range(self.n_tiles))
        plt.grid()
        plt.show()
