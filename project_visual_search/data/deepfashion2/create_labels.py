import json
import os
from collections import defaultdict

from PIL import Image
from source.utils import COCOParser, setup_google_credentials
from tqdm import tqdm


def create_labels(DATA_PATH, gcs=True):
    annotation_file = DATA_PATH + "instances.json"
    images_dir = DATA_PATH + "images/"
    labels_dir = DATA_PATH + "labels/"

    if not os.path.isdir(labels_dir):
        os.mkdir(labels_dir)

    coco = COCOParser(annotation_file, images_dir)
    total_images = len(coco.get_imgIds())  # total number of images
    img_ids = coco.get_imgIds()

    for i, im in tqdm(enumerate(img_ids), total=total_images):
        image = Image.open(f"{images_dir}{str(im).zfill(6)}.jpg")
        width, height = image.size
        ann_ids = coco.get_annIds(im)
        annotations = coco.load_anns(ann_ids)
        lines = [
            f"{anno['category_id']} {anno['bbox'][0] /width} {anno['bbox'][1]/height} {anno['bbox'][2]/width} {anno['bbox'][3]/height}"
            for anno in annotations
        ]

        if gcs:
            session = setup_google_credentials()
            local_path = f"{str(im).zfill(6)}.txt"
            with open(local_path, "w") as f:
                f.writelines("\n".join(lines))

            session = setup_google_credentials()
            # The bucket on GCS in which to write the CSV file
            bucket = session.gcs_client.get_bucket("cs-dio-euw4-dev")
            # The name assigned to the CSV file on GCS
            blob = bucket.blob(DATA_PATH + f"labels/{str(im).zfill(6)}.txt")
            blob.upload_from_filename(local_path)

        with open(f"{labels_dir}{str(im).zfill(6)}.txt", "w") as f:
            f.writelines("\n".join(lines))
