import argparse

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from datasets import load_dataset
from transformers import (AutoModelForObjectDetection, DetrImageProcessor,
                          Trainer, TrainingArguments)

torch.cuda.empty_cache()


def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i - 1],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i - 1]),
        }
        annotations.append(new_ann)

    return annotations


def main(args):
    fashionpedia = load_dataset("detection-datasets/fashionpedia_4_categories")
    categories = fashionpedia["train"].features["objects"].feature["category"].names

    id2label = {index: x for index, x in enumerate(categories, start=0)}
    label2id = {v: k for k, v in id2label.items()}

    checkpoint = "facebook/detr-resnet-50"
    image_processor = DetrImageProcessor.from_pretrained(checkpoint)

    def collate_fn(batch):
        pixel_values = [item["pixel_values"] for item in batch]
        encoding = image_processor.pad(pixel_values, return_tensors="pt")
        labels = [item["labels"] for item in batch]
        batch = {}
        batch["pixel_values"] = encoding["pixel_values"]
        batch["pixel_mask"] = encoding["pixel_mask"]
        batch["labels"] = labels
        return batch

    transform = transforms.Compose(
        [
            transforms.Resize((640, 640)),
            transforms.RandomHorizontalFlip(1),
            transforms.ColorJitter(contrast=0.5),
        ]
    )

    def transform_aug_ann(examples):
        """transforming a batch"""

        image_ids = examples["image_id"]
        images, bboxes, area, categories = [], [], [], []
        for image, objects in zip(examples["image"], examples["objects"]):
            image = np.array(image.convert("RGB"))[:, :, ::-1]
            im, bb, labels = transform(image, objects["bbox"], objects["category"])

            area.append(objects["area"])
            images.append(im)
            bboxes.append(bb)
            categories.append(labels)

        targets = [
            {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
            for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
        ]

        return image_processor(images=images, annotations=targets, return_tensors="pt")

    fashionpedia["train"] = fashionpedia["train"].with_transform(transform_aug_ann)

    model = AutoModelForObjectDetection.from_pretrained(
        pretrained_model_name_or_path="facebook/detr-resnet-50",
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        auto_find_batch_size=args.auto_find_batch_size,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        fp16=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_steps=50,
        learning_rate=args.lr,
        weight_decay=1e-4,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=fashionpedia["train"],
        tokenizer=image_processor,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        default="detr-resnet-50_finetuned_fashionpedia",
        type=str,
        help="Directory where to save logs",
    )

    parser.add_argument(
        "--epochs", default=50, type=int, help="Number of epochs of training"
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Mini batch size for stochastic gradient descent algorithms.",
    )
    parser.add_argument(
        "--auto_find_batch_size",
        default=False,
        type=bool,
        help="Whether to find a batch size that will fit into memory automatically through exponential decay, avoiding CUDA Out-of-Memory errors.",
    )
    parser.add_argument(
        "--lr", default=1e-5, type=float, help="Learning rate for the training"
    )

    parser.add_argument(
        "--save_total_limit",
        default=10,
        type=int,
        help="Total number of checkpoints to save during training",
    )

    args = parser.parse_args()

    main(args)
