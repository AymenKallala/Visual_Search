import argparse
import json
import os
from collections import defaultdict

from PIL import Image
from tqdm import tqdm


class COCOParser:
    def __init__(self, anns_file, imgs_dir):
        with open(anns_file, "r") as f:
            coco = json.load(f)

        self.annIm_dict = defaultdict(list)
        self.cat_dict = {}
        self.annId_dict = {}
        self.im_dict = {}
        self.licenses_dict = {}
        for ann in coco["annotations"]:
            self.annIm_dict[ann["image_id"]].append(ann)
            self.annId_dict[ann["id"]] = ann
        for img in coco["images"]:
            self.im_dict[img["id"]] = img
        for cat in coco["categories"]:
            self.cat_dict[cat["id"]] = cat

    def get_imgIds(self):
        return list(self.im_dict.keys())

    def get_annIds(self, im_ids):
        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        return [ann["id"] for im_id in im_ids for ann in self.annIm_dict[im_id]]

    def load_anns(self, ann_ids):
        im_ids = ann_ids if isinstance(ann_ids, list) else [ann_ids]
        return [self.annId_dict[ann_id] for ann_id in ann_ids]

    def load_cats(self, class_ids):
        class_ids = class_ids if isinstance(class_ids, list) else [class_ids]
        return [self.cat_dict[class_id] for class_id in class_ids]


def create_labels(DATA_PATH):
    annotation_file = DATA_PATH + "instances.json"
    images_dir = DATA_PATH + "images/"
    labels_dir = DATA_PATH + "labels/"

    if not os.path.isdir(labels_dir):
        os.mkdir(labels_dir)

    coco = COCOParser(annotation_file, images_dir)
    total_images = len(coco.get_imgIds())  # total number of images
    img_ids = coco.get_imgIds()

    for i, im in tqdm(enumerate(img_ids), total=total_images):
        im_object = coco.im_dict[im]
        width, height = im_object["width"], im_object["height"]
        ann_ids = coco.get_annIds(im)
        annotations = coco.load_anns(ann_ids)
        lines = [
            f"{anno['category_id']} {anno['bbox'][0] /width} {anno['bbox'][1]/height} {anno['bbox'][2]/width} {anno['bbox'][3]/height}"
            for anno in annotations
        ]

        with open(f"{labels_dir}{str(im)}.txt", "w") as f:
            f.writelines("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        default="/home/aka/visual-search/project-visual-search/datasets/fashionpedia/val/",
        type=str,
        help="Dataset to create labels .txt files for",
    )

    args = parser.parse_args()

    create_labels(args.data_path)
