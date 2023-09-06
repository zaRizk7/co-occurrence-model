import argparse
import logging
import os
import warnings
from math import log

import pandas as pd
import torch
import torch.nn.functional as F
from torch_optimizer import Lamb
from tqdm import tqdm

import utils
from made import *

parser = argparse.ArgumentParser(
    prog="MADETraining", description="Train MADE model for co-occurrence data"
)

parser.add_argument("--train-dir", type=str, help="Training data directory")
parser.add_argument("--test-dir", type=str, help="Validation data directory")
parser.add_argument(
    "--export-dir", type=str, help="Parameter and output export directory"
)
parser.add_argument(
    "--export-name", type=str, help="Parameter and output export filename"
)
parser.add_argument("--num-epochs", type=int, help="Number of epochs")
parser.add_argument(
    "--depth",
    type=int,
    help="Number of hidden layers (zero equates to a single hidden layer MADE)",
    default=0,
)
parser.add_argument(
    "--width", type=int, help="Number of neurons per hidden layer", default=256
)
parser.add_argument(
    "--activation",
    type=str,
    help="Activation function for hidden layer",
    default="relu",
)
parser.add_argument(
    "--bias",
    type=bool,
    help="Use bias for each layer",
    default=True,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--condition",
    type=bool,
    help="Use conditioning weights for each layer",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--one-hot-input",
    type=bool,
    help="Use one-hot encoded input for MADE.",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--batch-size", type=int, help="Batch size for SGD (full by default)", default=0
)
parser.add_argument(
    "--learning-rate", type=float, help="Learning rate for optimizer", default=1e-4
)
parser.add_argument(
    "--max-tolerance",
    type=int,
    help="Maximum tolerance limit for early stopping",
    default=20,
)
parser.add_argument(
    "--scheduler",
    type=bool,
    help="Automatically set a OneCycleLR scheduler for training",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--scheduler-max-learning-rate",
    type=float,
    help="Maximum learning rate for scheduler",
    default=1e-2,
)
parser.add_argument(
    "--lamb",
    type=bool,
    help="Replaces Adam optimizer with LAMB optimizer for large batch size training",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--num-sample-train",
    type=int,
    help="Number of sampling per training step",
    default=1,
)
parser.add_argument(
    "--num-sample-eval",
    type=int,
    help="Number of sampling per evaluation step",
    default=100,
)
parser.add_argument(
    "--device",
    type=str,
    help="Device for training",
    choices=["cpu", "cuda"],
    default="cpu",
)
parser.add_argument(
    "--num-workers", type=int, help="Number of workers for dataloader", default=0
)
parser.add_argument("--random-seed", type=int,
                    help="Seed for reproduction", default=0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    torch.manual_seed(args.random_seed)
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)

    if not os.path.exists(args.train_dir):
        raise ValueError("train_dir does not exists!")

    if not os.path.exists(args.test_dir):
        raise ValueError("test_dir does not exists!")

    if args.num_epochs <= 0:
        raise ValueError("num_epochs must be atleast equal to one!")

    if args.depth < 0:
        raise ValueError("depth must be atleast zero!")

    if args.width <= 0:
        raise ValueError("width must be a positive non-zero value!")

    if args.activation not in ["relu", "gelu", "identity", "tanh", "sigmoid"]:
        raise ValueError(
            "activation must be either relu, gelu, identity, tanh, or sigmoid!"
        )

    if args.learning_rate < 0:
        raise ValueError("learning_rate must be positive non-zero value!")

    if not torch.cuda.is_available():
        args.device = "cpu"
        raise RuntimeWarning("cuda is not available, cpu is set as device!")

    if args.num_sample_train <= 0:
        raise ValueError("num_sample_train must be a non-zero positive value!")

    if args.num_sample_eval <= 0:
        raise ValueError("num_sample_eval must be a non-zero positive value!")

    dataset = dict(
        train=utils.dataset.ObjectCooccurrenceCOCODataset(args.train_dir),
        test=utils.dataset.ObjectCooccurrenceCOCODataset(args.test_dir),
    )

    if args.batch_size <= 0:
        args.batch_size = len(dataset)
        warnings.warn(
            "invalid batch_size, setting batch_size to the full data size!",
            RuntimeWarning,
        )

    Activation = torch.nn.Sigmoid
    if args.activation == "relu":
        Activation = torch.nn.ReLU
    if args.activation == "gelu":
        Activation = torch.nn.GELU
    if args.activation == "identity":
        Activation = torch.nn.Identity
    if args.activation == "tanh":
        Activation = torch.nn.Tanh

    dataloader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )

    dataloader = dict(
        train=torch.utils.data.DataLoader(
            dataset["train"], shuffle=True, **dataloader_args
        ),
        test=torch.utils.data.DataLoader(
            dataset["test"], shuffle=False, **dataloader_args
        ),
    )

    max_value = dataset["train"].features.to_numpy().max()

    layers = []

    if args.one_hot_input:
        layers.append(
            MaskedAutoregressiveLinear(
                in_features=dataset["train"].features.shape[1],
                out_features=args.width,
                in_dims=max_value + 1,
                bias=args.bias,
                weight_condition=args.weight_,
            )
        )
    else:
        layers.append(
            MaskedAutoregressiveLinear(
                in_features=dataset["train"].features.shape[1],
                out_features=args.width,
                bias=args.bias,
                weight_condition=args.condition,
            )
        )
    layers.append(Activation())

    for _ in range(args.depth):
        layers.append(
            MaskedAutoregressiveLinear(
                in_features=args.width,
                out_features=args.width,
                bias=args.bias,
                weight_condition=args.condition,
            )
        )
        layers.append(Activation())

    layers.append(
        MaskedAutoregressiveLinear(
            in_features=args.width,
            out_features=dataset["train"].features.shape[1],
            out_dims=max_value + 1,
            bias=args.bias,
            weight_condition=args.condition,
        )
    )

    made = MADE(*layers).to(args.device)

    Optimizer = torch.optim.Adam
    if args.lamb:
        Optimizer = Lamb

    opt = Optimizer(made.parameters(), args.learning_rate)

    sch = None
    if args.scheduler:
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            args.scheduler_max_learning_rate,
            epochs=args.num_epochs,
            steps_per_epoch=len(dataloader["train"]),
        )

    print(made)
    print(f"Total parameters: {sum(p.numel() for p in made.parameters()):,}")

    history = dict(train=[], test=[])
    grad_norm = []
    with tqdm(total=args.num_epochs, unit="epoch") as pbar:
        tolerance = 1
        loss_best = torch.inf
        for _ in range(args.num_epochs):
            pbar.set_description(
                f"[patience: {tolerance}/{args.max_tolerance}]")

            grad_norm.extend(train_one_epoch(
                dataloader['train'], made, opt, sch, args.num_sample_train, args.one_hot_input, args.device, max_value+1))
            history["train"].append(-log_likelihood(
                dataloader['train'], made, args.num_sample_eval, args.one_hot_input, args.device, max_value+1))
            history["test"].append(-log_likelihood(
                dataloader['test'], made, args.num_sample_eval, args.one_hot_input, args.device, max_value+1))

            history["train"][-1] /= len(dataset["train"])
            history["test"][-1] /= len(dataset["test"])

            pbar.set_postfix(
                [
                    ("train_nll", f'{history["train"][-1]:.4f}'),
                    ("test_nll", f'{history["test"][-1]:.4f}'),
                ]
            )
            pbar.update()

            if loss_best > history["test"][-1]:
                tolerance = 1
                loss_best = history["test"][-1]
            else:
                tolerance += 1

            if tolerance > args.max_tolerance:
                break

    os.makedirs(args.export_dir, exist_ok=True)
    df = pd.DataFrame(history)
    df.index += 1
    df.to_csv(f"{args.export_dir}/{args.export_name}-history.csv",
              index_label="epoch")
    df = pd.DataFrame(grad_norm)
    df.index += 1
    df.to_csv(f"{args.export_dir}/{args.export_name}-grad_norm.csv")
    torch.save(made, f"{args.export_dir}/{args.export_name}-model.pt")
