import argparse
import logging
import os
import pickle
import warnings
from collections import OrderedDict

import pandas as pd
import torch
from tqdm import tqdm

import einet_addons
import utils
from EinsumNetwork import EinsumNetwork

parser = argparse.ArgumentParser(
    prog="MixtureTrainer", description="Train mixture model for co-occurrence data"
)

parser.add_argument("--train-dir", type=str, help="Training data directory")
parser.add_argument("--valid-dir", type=str, help="Validation data directory")
parser.add_argument(
    "--structure-dir", type=str, help="Pickled structure (nx.DiGraph) directory."
)
parser.add_argument(
    "--export-dir", type=str, help="Parameter and output export directory"
)
parser.add_argument(
    "--export-name", type=str, help="Parameter and output export filename"
)
parser.add_argument("--num-input-distributions", type=int, help="Number of mixtures")
parser.add_argument("--num-sums", type=int, help="Number of sums")
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
    "--update-frequency",
    type=int,
    help="EM update frequency for stochastic EM",
    default=1,
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

    if not os.path.exists(args.structure_dir):
        raise ValueError("structure_dir does not exists!")

    if args.num_input_distributions <= 0:
        args.num_input_distributions = 1
        warnings.warn(
            "num_input_distributions must be a positive greater than zero! Setting it to one!",
            RuntimeWarning,
        )

    if args.num_sums <= 0:
        args.num_sums = 1
        warnings.warn(
            "num_sums must be a positive greater than zero! Setting it to one!",
            RuntimeWarning,
        )

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

    with open(args.structure_dir, "rb") as f:
        region_graph = pickle.load(f)

    args_einet = EinsumNetwork.Args(
        num_classes=1,
        num_input_distributions=args.num_input_distributions,
        exponential_family=EinsumNetwork.CategoricalArray,
        exponential_family_args={"K": dataset_train.features.to_numpy().max() + 1},
        num_sums=args.num_sums,
        num_var=dataset_train.features.shape[1],
        online_em_frequency=args.update_frequency,
        online_em_stepsize=args.step_size,
    )

    einet = einet_addons.EiNetForest(region_graph, args_einet)
    einet.initialize()
    einet = einet.to(args.device)

    print(f"Total parameters: {sum(p.numel() for p in einet.parameters()):,}")

    history = []
    grad_norm = []
    tolerance = 1
    avg_nll_valid_best = torch.inf
    for epoch in (pbar := tqdm(range(args.num_iterations), unit="epoch", leave=True)):
        pbar.set_description(f"[patience: {tolerance}/{args.max_tolerance}]")

        grad_norm.extend(
            einet_addons.train_one_epoch(dataloader_train, einet, args.device)
        )

        ll_train = einet_addons.log_likelihood(dataloader_train2, einet, args.device)

        ll_valid = einet_addons.log_likelihood(dataloader_valid, einet, args.device)

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
            tolerance = 1
        else:
            tolerance += 1

        if tolerance > args.max_tolerance:
            break

        pbar.set_postfix([(k, f"{v:.4f}") for k, v in history[-1].items()])

    os.makedirs(args.export_dir, exist_ok=True)
    df = pd.DataFrame(history)
    df.index += 1
    df.to_csv(f"{args.export_dir}/{args.export_name}-history.csv", index_label="epoch")
    df = pd.DataFrame(grad_norm)
    df.index += 1
    df.to_csv(
        f"{args.export_dir}/{args.export_name}-grad_norm.csv",
        index_label="minibatch_step",
    )
    torch.save(
        {
            "params": einet.state_dict(),
            "args": args_einet,
            "region_graph": region_graph,
        },
        f"{args.export_dir}/{args.export_name}-params.pt",
    )
