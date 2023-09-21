import argparse

from ultralytics import YOLO, checks


def main(args):
    checks()

    yolo_v8 = YOLO(args.model)
    yolo_v8.train(
        data=args.data_path,
        batch=args.batch_size,
        epochs=args.epochs,
        pretrained=True,
        plots=args.plots,
        val=True,
        lr0=args.lr,
        momentum=args.momentum,
        save_period=args.save_period,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="../data/configs/fashionpedia.yaml",
        type=str,
        help="Path for the data config file .yaml",
    )
    parser.add_argument(
        "--epochs", default=50, type=int, help="Number of epochs of training"
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Mini batch size for stochastic gradient descent algorithms.",
    )
    parser.add_argument(
        "--plots", default=True, type=bool, help=" Generate results plots or not."
    )
    parser.add_argument(
        "--model", default="yolov8n.pt", type=str, help="Version of YOLO to train."
    )
    parser.add_argument(
        "--lr", default=0.01, type=float, help="Learning rate for the training"
    )
    parser.add_argument(
        "--momentum", default=0.937, type=float, help="Momentum for the training"
    )
    parser.add_argument(
        "--save_period",
        default=-1,
        type=int,
        help="Mini batch size for stochastic gradient descent algorithms.",
    )

    args = parser.parse_args()

    main(args)
