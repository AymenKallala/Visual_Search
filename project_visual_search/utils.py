import os
import glob
import json
from collections import defaultdict


def setup_google_credentials():
    os.environ[
        "GOOGLE_APPLICATION_CREDENTIALS"
    ] = "/home/dataiku/gcp_auth/ai-factory-dev.json"
    os.environ["GS_BUCKET_NAME"] = "gs://cs-dio-euw4-dev"
    os.environ["GS_BASE_LOCATION"] = "dataiku/"
    os.environ["DATABASE_NAME"] = "VISUAL_SEARCH_EXPLORATION"
    os.environ["DATASET_ID"] = "cdc_dev_dio_bqd_dss"
    from tooling_aif_bigquery.base import get_default_session

    session = get_default_session()

    return session


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
        for license in coco["licenses"]:
            self.licenses_dict[license["id"]] = license

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

    def get_imgLicenses(self, im_ids):
        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        lic_ids = [self.im_dict[im_id]["license"] for im_id in im_ids]
        return [self.licenses_dict[lic_id] for lic_id in lic_ids]
