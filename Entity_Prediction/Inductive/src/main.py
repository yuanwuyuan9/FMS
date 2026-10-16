import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch

from data_loader import load_data
from train import train


def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_setting(args):
    print()
    print("=============================================")
    print("          Inductive Entity Prediction         ")
    print("=============================================")
    print(f"dataset: {args.dataset}")
    print(f"epoch: {args.epoch}")
    print(f"batch_size: {args.batch_size}")
    print(f"dim: {args.dim}")
    print(f"l2: {args.l2}")
    print(f"lr: {args.lr}")
    print(f"seed: {args.seed}")
    print(f"feature_type: {args.feature_type}")
    print(f"context_hops: {args.context_hops}")
    print(f"neighbor_samples: {args.neighbor_samples}")
    print(f"neighbor_agg: {args.neighbor_agg}")
    print(f"tok_k: {args.tok_k}")
    print(f"num_attention_heads: {args.num_heads}")
    print(f"eval_query_batch: {args.eval_query_batch}")
    print(f"select_split: {args.select_split}")
    print()
    print("=============================================")
    print("              CFM Module Settings             ")
    print("=============================================")
    print(f"CFM Variant: {args.cfm_variant}")
    print(f"CFM MLP Dimension: {args.cfm_model_dim}")
    print(f"CFM Sigma: {args.cfm_sigma}")
    if args.cfm_variant == "OTCFM":
        print(f"CFM OT Method: {args.cfm_ot_method}")
    print(f"inference_t_strategy: {args.inference_t_strategy}")
    print(f"inference_t: {args.inference_t}")
    print(f"inference_mc_samples: {args.inference_mc_samples}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Inductive FMS entity prediction")
    parser.add_argument("--cuda", help="use gpu", default=False, action="store_true")
    parser.add_argument("--dataset", type=str, required=True, help="base dataset name, e.g. nell_v1")
    parser.add_argument("--epoch", type=int, default=20, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="query batch size for training")
    parser.add_argument("--dim", type=int, default=64, help="hidden dimension")
    parser.add_argument("--l2", type=float, default=1e-7, help="l2 regularization weight")
    parser.add_argument("--lr", type=float, default=5e-3, help="learning rate")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--feature_type", type=str, default="id", help="type of relation features")

    parser.add_argument("--context_hops", type=int, default=2, help="number of context hops")
    parser.add_argument("--neighbor_samples", type=int, default=16, help="number of sampled neighbors for one hop")
    parser.add_argument(
        "--neighbor_agg",
        type=str,
        default="concat",
        choices=["mean", "concat", "cross", "attention"],
        help="neighbor aggregator",
    )
    parser.add_argument("--tok_k", type=int, default=8, help="top-k value")
    parser.add_argument("--num_heads", type=int, default=4, help="number of attention heads")
    parser.add_argument("--eval_query_batch", type=int, default=8, help="query batch size during ranking")
    parser.add_argument(
        "--select_split",
        type=str,
        default="base-valid",
        choices=["base-valid", "ind-valid"],
        help="validation split used to choose the final reported checkpoint",
    )

    parser.add_argument("--lamda", type=float, default=1.2, help="CFM loss weight")
    parser.add_argument("--cfm_model_dim", type=int, default=128, help="dimension of the CFM MLP")
    parser.add_argument(
        "--cfm_variant",
        type=str,
        default="OTCFM",
        choices=["CFM", "OTCFM", "TargetCFM", "SBCFM", "VPCFM"],
        help="variant of Conditional Flow Matching to use",
    )
    parser.add_argument("--cfm_sigma", type=float, default=0.1, help="sigma value for CFM")
    parser.add_argument("--cfm_ot_method", type=str, default="exact", help="OT method for SBCFM")
    parser.add_argument(
        "--inference_t_strategy",
        type=str,
        default="fixed",
        choices=["random", "fixed", "mc"],
        help="pseudo-time strategy used during inference",
    )
    parser.add_argument("--inference_t", type=float, default=0.5, help="fixed pseudo-time value")
    parser.add_argument("--inference_mc_samples", type=int, default=20, help="MC samples for inference")

    args = parser.parse_args()
    if args.dataset.endswith("_ind"):
        original = args.dataset
        args.dataset = args.dataset[:-4]
        print(f"Received {original}; using base dataset {args.dataset} with {original} as inductive graph.")

    if args.cuda and not torch.cuda.is_available():
        print("CUDA requested but not available; using CPU")
        args.cuda = False
    print("Using CUDA" if args.cuda else "Using CPU")

    set_random_seed(args.seed)
    print_setting(args)
    data = load_data(args)
    print("Current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    train(args, data)


if __name__ == "__main__":
    main()
