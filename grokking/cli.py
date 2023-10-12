from argparse import ArgumentParser

from data import ALL_OPERATIONS
from training import main

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--operation", type=str, choices=ALL_OPERATIONS.keys(), default="x/y")
    parser.add_argument("--training_fraction", type=float, default=0.8)
    parser.add_argument("--prime", type=int, default=37)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_model", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.2)
    parser.add_argument("--num_steps", type=int, default=2e4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(args)
