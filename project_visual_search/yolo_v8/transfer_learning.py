import argparse

from ultralytics import YOLO, checks


def main(args):
    checks()

    def freeze_layer(trainer):
        model = trainer.model
        num_freeze = args.freeze  # Number of layers we want to freeze
        freezed_p = 0

        ss = 0
        for k, v in model.named_parameters():
            nn = 1
            for s in v.size():
                nn = nn * s
            ss += nn
        print(f"The model contains {ss} parameters in total")

        print(f"Freezing {num_freeze} layers")
        freeze = [f"model.{x}." for x in range(num_freeze)]  # layers to freeze
        for k, v in model.named_parameters():
            if any(x in k for x in freeze):
                print(f"freezing {k}")
                v.requires_grad = False
                nn = 1
                for s in v.size():
                    nn = nn * s
                freezed_p += nn
        print(f"{num_freeze} layers are freezed.")
        print(f"{freezed_p} parameters are freezed.")
        print(f"Still have {ss - freezed_p} parameters trainable")

    yolo_v8 = YOLO(args.model)
    yolo_v8.add_callback("on_train_start", freeze_layer)
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
        "--freeze",
        default=22,
        type=int,
        help="Number of blocks to freeze in the model",
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
