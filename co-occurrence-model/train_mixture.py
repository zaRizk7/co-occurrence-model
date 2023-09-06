import argparse
import logging
import os
import warnings
from collections import OrderedDict
from copy import deepcopy

import pandas as pd
import torch
from tqdm import tqdm

import mixture
import utils

parser = argparse.ArgumentParser(
    prog="MixtureTrainer", description="Train mixture model for co-occurrence data"
)

parser.add_argument("--train-dir", type=str, help="Training data directory")
parser.add_argument("--valid-dir", type=str, help="Validation data directory")
parser.add_argument(
    "--export-dir", type=str, help="Parameter and output export directory"
)
parser.add_argument(
    "--export-name", type=str, help="Parameter and output export filename"
)
parser.add_argument("--num-mixtures", type=int, help="Number of mixtures")
parser.add_argument("--num-restarts", type=int, help="Number of restarts", default=10)
parser.add_argument(
    "--num-iterations", type=int, help="Maximum number of iteration", default=100
)
parser.add_argument(
    "--batch-size",
    type=int,
    help="Batch size for stochastic EM (full by default)",
    default=0,
)
parser.add_argument(
    "--step-size",
    type=float,
    help="Step size for stochastic EM parameter update",
    default=1e-3,
)
parser.add_argument(
    "--max-tolerance",
    type=int,
    help="Maximum tolerance limit for early stopping",
    default=10,
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
parser.add_argument("--random-seed", type=int, help="Seed for reproduction", default=0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    torch.manual_seed(args.random_seed)
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)

    if not os.path.exists(args.train_dir):
        raise ValueError("train_dir does not exists!")

    if not os.path.exists(args.valid_dir):
        raise ValueError("valid_dir does not exists!")

    if args.num_mixtures <= 0:
        raise ValueError("num_mixture must be a positive greater than zero!")

    if args.num_restarts <= 0:
        raise ValueError("num_restarts must be a positive greater than zero!")

    if args.num_iterations <= 0:
        raise ValueError("num_iteration must be a positive greater than zero!")

    if not torch.cuda.is_available():
        args.device = "cpu"
        raise RuntimeWarning("cuda is not available, cpu is set as device!")

    invalid_step_size = not (0 <= args.step_size <= 1.0)

    dataset_train = utils.dataset.ObjectCooccurrenceCOCODataset(args.train_dir)
    dataset_valid = utils.dataset.ObjectCooccurrenceCOCODataset(args.valid_dir)

    if args.batch_size <= 0:
        args.batch_size = len(dataset_train)
        warnings.warn(
            "invalid batch_size, setting batch_size to the full data size!",
            RuntimeWarning,
        )

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=invalid_step_size,
        pin_memory=args.device == "cuda",
    )

    dataloader_train2 = torch.utils.data.DataLoader(
        dataset_train,
        args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )

    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )

    if invalid_step_size:
        args.step_size = 1 / len(dataloader_train)
        warnings.warn(
            f"invalid step_size, setting step_size to 1/{len(dataloader_train)}!",
            RuntimeWarning,
        )

    if args.max_tolerance <= 0:
        args.max_tolerance = args.num_iterations
        warnings.warn(
            "Invalid max_tolerance, setting the tolerance equal to num_iterations!"
        )

    avg_nll_valid_best_restarts = torch.full([], float("inf"))
    history_best_restarts = float("inf")
    params_best_restarts = None

    for n_res in (pbar1 := tqdm(range(args.num_restarts), unit="restart")):
        params = mixture.initialize_parameters(dataset_train, args.num_mixtures).to(
            args.device
        )

        model = mixture.CategoricalMixture(params, args.step_size)
        model = model.to(args.device)

        history = []
        tolerance = 1
        avg_nll_valid_best = float("inf")
        for i in (
            pbar2 := tqdm(range(args.num_iterations), unit="iteration", leave=True)
        ):
            pbar2.set_description(f"[restart {n_res + 1}/{args.num_restarts}]")

            with torch.no_grad():
                for inputs in dataloader_train:
                    inputs = inputs.to(args.device)
                    model.em_step(inputs)

            ll_train = mixture.log_likelihood(dataloader_train2, model, args.device)
            ll_valid = mixture.log_likelihood(dataloader_valid, model, args.device)

            history.append(
                OrderedDict(
                    [
                        ("avg_nll_train", -ll_train / len(dataset_train)),
                        ("avg_nll_valid", -ll_valid / len(dataset_valid)),
                    ]
                )
            )

            if avg_nll_valid_best > history[-1]["avg_nll_valid"]:
                avg_nll_valid_best = history[-1]["avg_nll_valid"]
                history_best = deepcopy(history)
                params_best = deepcopy(model.params)
            else:
                tolerance += 1

            if tolerance > args.max_tolerance:
                break

            pbar2.set_postfix(
                [
                    ("avg_nll_train", f'{history[-1]["avg_nll_train"]:.4f}'),
                    ("avg_nll_valid", f'{history[-1]["avg_nll_valid"]:.4f}'),
                    ("avg_nll_valid_best", f"{avg_nll_valid_best:.4f}"),
                ]
            )

        if avg_nll_valid_best_restarts > avg_nll_valid_best:
            avg_nll_valid_best_restarts = avg_nll_valid_best
            history_best_restarts = deepcopy(history_best)
            params_best_restarts = deepcopy(params_best)

        pbar1.set_postfix(
            [
                ("avg_nll_valid_current", f"{avg_nll_valid_best:.4f}"),
                ("avg_nll_valid_best", f"{avg_nll_valid_best_restarts:.4f}"),
            ]
        )

    os.makedirs(args.export_dir, exist_ok=True)
    df = pd.DataFrame(history_best_restarts)
    df.index += 1
    df.to_csv(
        f"{args.export_dir}/{args.export_name}-history.csv", index_label="iteration"
    )
    torch.save(params_best_restarts, f"{args.export_dir}/{args.export_name}-params.pt")
