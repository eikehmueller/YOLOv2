from abc import ABC, abstractmethod
import glob
import xml.etree.ElementTree as ET
from pycocotools.coco import COCO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl


class ImageReader(ABC):
    """Abstract base class for reading annotated images from a dataset

    This class provides a common interface for reading images from a dataset and generating
    bespoke annotations for object detection.
    """

    def __init__(self, image_size=416, n_tiles=13):
        """Create a new instance

        :arg image_size: size of resized images
        :arg n_tiles: number of tiles each dimension is subdivided into

        self.image_size = image_size
        self.n_tiles = n_tiles
        """
        self.image_size = image_size
        self.n_tiles = n_tiles
        self.class_category_name_map = {}

    @abstractmethod
    def get_image_ids(self, category_names=None):
        """Return the ids of all images in the dataset which contain objects that belong
        to certain categories

        :arg category_names: names of categories that should appear in the images.
                             If None, use any category.
        """

    def get_n_images(self, category_names=None):
        """Get the number of images which belong to a certain category

        :arg category_names: names of categories that should appear in the images.
                             If None, use any category.
        """
        return len(self.get_image_ids(category_names))

    @abstractmethod
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


class COCOImageReader(ImageReader):
    """Class for reading and resizing images with bbox annotations from COCO dataset

    This class allows reading images from the COCO-dataset and generating bespoke
    annotations for object detection.
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
        super().__init__(image_size, n_tiles)
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
        """Return the ids of all images in the dataset which contain objects that belong
        to certain categories

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
        image = (
            np.asarray(
                cv2.resize(
                    cv2.imread(self.image_dir + "/" + img["file_name"]),
                    (self.image_size, self.image_size),
                ),
                dtype=np.float32,
            )
            / 255.0
        )
        image = image[:, :, ::-1]
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


class PascalVOCImageReader(ImageReader):
    """Class for reading and resizing images with bbox annotations from Pascal VOC dataset

    This class allows reading images from the PascalVOC-dataset and generating bespoke
    annotations for object detection.
    """

    def __init__(
        self,
        data_dir="../../pascalvocdata/VOC2012/",
        image_size=416,
        n_tiles=13,
        verbose=False,
    ):
        """Create new instance

        :arg data_dir: directory containing the data. This is assumed to contain the
                       subdirectories /Annotations and /JPEGImages
        :arg image_size: size of resized images
        :arg n_tiles: number of tiles each dimension is subdivided into
        :arg verbose: print additional information
        """
        super().__init__(image_size, n_tiles)
        self.data_dir = data_dir
        annotation_dir = self.data_dir + "/Annotations/"
        annotation_filenames = glob.glob(annotation_dir + "/*.xml")
        self.image_ids = list(range(len(annotation_filenames)))
        self.all_category_names = set()
        self.annotations = []
        self.image_ids = {}
        for idx, filename in enumerate(annotation_filenames):
            annotation = self._extract_raw_bboxes(filename)
            self.annotations.append(annotation)
            for raw_bbox in annotation["raw_bboxes"]:
                category_name = raw_bbox["category_name"]
                if category_name not in self.image_ids.keys():
                    self.image_ids[category_name] = set()
                self.image_ids[category_name].add(idx)
        # Map from classes (=class indices) to category names
        self.class_category_name_map = {
            j: x for j, x in enumerate(self.all_category_names)
        }
        # Inverse map
        self.category_name_class_map = {
            x: j for j, x in enumerate(self.all_category_names)
        }
        self.n_classes = len(self.all_category_names)
        print(f"number of classes = {self.n_classes}")
        if verbose:
            print(
                "Pascal VOC categories: \n{}\n".format(
                    ", ".join(self.all_category_names)
                )
            )

    def _extract_raw_bboxes(self, filename):
        """Extract bounding box information from a single xml file

        :arg filename: name of xml file to parse
        """
        tree = ET.parse(filename)
        root = tree.getroot()
        # extract name of file
        node_filename = root.find("filename")
        filename = node_filename.text
        image_size = {}
        for node_size_spec in root.find("size"):
            image_size[node_size_spec.tag] = int(node_size_spec.text)
        # extract raw bounding boxes
        raw_bboxes = []
        for node_object in root.iter("object"):
            bbox = {}
            category_name = node_object.find("name").text
            self.all_category_names.add(category_name)
            bbox["category_name"] = category_name
            for node_bbox in node_object.find("bndbox"):
                bbox[node_bbox.tag] = int(float(node_bbox.text))
            raw_bboxes.append(bbox)
        return {
            "image_filename": filename,
            "image_width": image_size["width"],
            "image_height": image_size["height"],
            "raw_bboxes": raw_bboxes,
        }

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
        annotation = self.annotations[img_id]
        image_width = annotation["image_width"]
        image_height = annotation["image_height"]
        filename = self.data_dir + "/JPEGImages/" + annotation["image_filename"]
        image = (
            np.asarray(
                cv2.resize(
                    cv2.imread(filename),
                    (self.image_size, self.image_size),
                ),
                dtype=np.float32,
            )
            / 255.0
        )
        image = image[:, :, ::-1]
        bboxes = []
        for annotation in annotation["raw_bboxes"]:
            class_id = self.category_name_class_map[annotation["category_name"]]
            xmin = annotation["xmin"]
            xmax = annotation["xmax"]
            ymin = annotation["ymin"]
            ymax = annotation["ymax"]
            bboxes.append(
                {
                    "class": class_id,
                    "xc": round(self.image_size / image_width * 0.5 * (xmin + xmax)),
                    "yc": round(self.image_size / image_height * 0.5 * (ymin + ymax)),
                    "width": round(self.image_size / image_width * (xmax - xmin)),
                    "height": round(self.image_size / image_height * (ymax - ymin)),
                }
            )
        return {"image": image, "bboxes": bboxes}

    def get_image_ids(self, category_names=None):
        """Return the ids of all images in the dataset which contain objetcs that belong
        to a certain category

        :arg category_names: names of categories that should appear in the images.
                             If None, use any category.
        """
        ids = set(range(len(self.annotations)))
        if category_names is not None:
            for category_name in category_names:
                assert category_name in self.all_category_names
                ids = ids.intersection(self.image_ids[category_name])
        return list(ids)