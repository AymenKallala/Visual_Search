import argparse

from ultralytics import RTDETR


def eval(args):
    print("-------VALIDATION OF THE BEST MODEL --------")

    best_model_path = f"../../runs/detect/{args.run}/weights/best.pt"
    best_model = RTDETR(best_model_path)
    best_model.val(data=args.data, conf=0.25, plots=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        default="medium_deepfashion",
        type=str,
        help="Name of the model to evaluate in the runs directory",
    )
    parser.add_argument(
        "--data",
        default=None,
        type=str,
        help="Path to the data to evaluate on",
    )

    args = parser.parse_args()

    eval(args)
