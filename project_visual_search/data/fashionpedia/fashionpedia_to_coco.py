import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm


# format annotations the same as for training, no need for data augmentation
def formatted_anns(image_id, objects):
    annotations = []
    for i in range(len(objects["bbox_id"])):
        x, y, w, h = tuple(objects["bbox"][i])
        bbox = [x, y, w - x, h - y]
        new_ann = {
            "id": objects["bbox_id"][i],
            "category_id": objects["category"][i],
            "iscrowd": 0,
            "image_id": image_id,
            "area": objects["area"][i],
            "bbox": bbox,
        }
        annotations.append(new_ann)

    return annotations


# Save images and annotations into the files torchvision.datasets.CocoDetection expects


def save_anotationfile_and_images(fashionpedia, data_path, set, id2label):
    output_json = {}
    path_output = f"{data_path}/{set}"
    images_path = path_output + "/images"

    if not os.path.exists(path_output):
        os.makedirs(path_output)
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    path_anno = os.path.join(path_output, "instances.json")
    categories_json = [
        {"supercategory": "none", "id": id, "name": id2label[id]} for id in id2label
    ]
    output_json["images"] = []
    output_json["annotations"] = []
    print("GENERATING JSON FILE WITH ANNOTATIONS ")
    for example in tqdm(fashionpedia, total=fashionpedia.num_rows):
        ann = formatted_anns(example["image_id"], example["objects"])
        output_json["images"].append(
            {
                "id": example["image_id"],
                "width": example["image"].width,
                "height": example["image"].height,
                "file_name": f"{example['image_id']}.png",
            }
        )
        output_json["annotations"].extend(ann)
    output_json["categories"] = categories_json

    with open(path_anno, "w") as file:
        json.dump(output_json, file, ensure_ascii=False, indent=4)

    print("SAVING IMAGES IN IMAGE FOLDER")

    for data in tqdm(fashionpedia.iter(100), total=fashionpedia.num_rows / 100):
        for im, img_id in zip(data["image"], data["image_id"]):
            path_img = os.path.join(images_path, f"{img_id}.png")
            im.save(path_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        default="train",
        type=str,
        help="Dataset to create json instances for",
    )
    parser.add_argument(
        "--set", default="train", type=str, help="Dataset to create json instances for"
    )

    args = parser.parse_args()

    fashionpedia = load_dataset("detection-datasets/fashionpedia_4_categories")
    categories = fashionpedia[args.set].features["objects"].feature["category"].names

    id2label = {index: x for index, x in enumerate(categories, start=0)}
    label2id = {v: k for k, v in id2label.items()}

    save_anotationfile_and_images(
        fashionpedia[args.set], args.data_path, args.set, id2label
    )
