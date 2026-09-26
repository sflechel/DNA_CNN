from src.training import train
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Train the DNA CNN to find active regions in noncoding DNA"
    )
    parser.add_argument(
        "-ne",
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs spent training",
    )
    parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        default=16,
        help="CPU workers to assign to training",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=256,
        help="How many training samples to compute per batch",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.001,
        help="Rate at which the weights are updated",
    )
    parser.add_argument(
        "-l1",
        "--weight_decay",
        type=float,
        default=5e-7,
        help="Model weight decay parameter",
    )
    parser.add_argument(
        "-l2",
        "--output_decay",
        type=float,
        default=1e-8,
        help="Parameter of L1 penalty for output of fully connected layer",
    )
    parser.add_argument(
        "-l3",
        "--neuron_norm_max",
        type=float,
        default=0.9,
        help="Max norm for any neuron weight tensor",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
