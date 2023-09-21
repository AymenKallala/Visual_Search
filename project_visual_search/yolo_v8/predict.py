import argparse
import os
from glob import glob, iglob

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
from ultralytics import YOLO


def predict_img(model, img_path, conf, output_dir):
    image = Image.open(img_path)
    image = ImageOps.exif_transpose(image)
    results = model.predict(image, conf=conf, verbose=False)[0]

    preds = results.boxes.xywh
    id2label = results.names
    classes = results.boxes.cls
    probs = results.boxes.conf

    f, ax = plt.subplots()

    for i, pred in enumerate(preds):
        x_pred, y_pred, w_pred, h_pred = pred.cpu()
        rect_pred = plt.Rectangle(
            (x_pred, y_pred),
            w_pred,
            h_pred,
            linewidth=2,
            linestyle="--",
            edgecolor="blue",
            facecolor="none",
        )
        t_box = ax.text(
            x_pred,
            y_pred,
            f"{id2label[classes[i].item()]} : {np.around(probs[i].item(),decimals = 2)} ",
            color="white",
            fontsize=14,
        )
        t_box.set_bbox(
            dict(
                boxstyle="square, pad=0", facecolor="blue", alpha=0.6, edgecolor="blue"
            )
        )
        ax.add_patch(rect_pred)

    ax.axis("off")
    ax.imshow(image)
    ax.set_xlabel("Longitude")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{os.path.basename(img_path)}")
    plt.close()


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model_path = f"../../runs/detect/{args.run}/weights/best.pt"
    model = YOLO(model_path)

    img_paths = glob(f"{args.data_path}/*")

    for img_p in tqdm(img_paths, total=len(img_paths)):
        predict_img(model, img_p, args.conf, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        default="medium_deepfashion",
        type=str,
        help="Name of the model to evaluate in the runs directory",
    )
    parser.add_argument(
        "--data_path",
        default=None,
        type=str,
        help="Path to the data to evaluate on",
    )
    parser.add_argument(
        "--conf",
        default=0.25,
        type=float,
        help="Confiance threshold",
    )
    parser.add_argument(
        "--output_dir",
        default="/project-visual-search/runs/predict/test",
        type=str,
        help="Directory where to store predictions",
    )

    args = parser.parse_args()

    main(args)
