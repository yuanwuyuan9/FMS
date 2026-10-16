import os
import shlex
import subprocess
import sys


BASE_COMMAND = [sys.executable, "main.py"]
USE_CUDA_FOR_ALL = False


COMMON_CONFIG = {
    "--seed": "42",
    "--cfm_model_dim": "128",
    "--cfm_variant": "OTCFM",
    "--cfm_sigma": "0.1",
    "--cfm_ot_method": "exact",
    "--inference_t_strategy": "fixed",
}


DATASET_CONFIGS = {
    "WN18RR_v1": {
        "--epoch": "20",
        "--batch_size": "64",
        "--dim": "64",
        "--l2": "1e-7",
        "--lr": "5e-3",
        "--context_hops": "3",
        "--neighbor_samples": "8",
        "--neighbor_agg": "attention",
        "--tok_k": "4",
        "--eval_query_batch": "8",
    },
    "WN18RR_v2": {
        "--epoch": "20",
        "--batch_size": "64",
        "--dim": "64",
        "--l2": "1e-7",
        "--lr": "5e-3",
        "--context_hops": "3",
        "--neighbor_samples": "8",
        "--neighbor_agg": "attention",
        "--tok_k": "4",
        "--eval_query_batch": "8",
    },
    "WN18RR_v3": {
        "--epoch": "20",
        "--batch_size": "64",
        "--dim": "64",
        "--l2": "1e-7",
        "--lr": "5e-3",
        "--context_hops": "3",
        "--neighbor_samples": "8",
        "--neighbor_agg": "attention",
        "--tok_k": "4",
        "--eval_query_batch": "8",
    },
    "WN18RR_v4": {
        "--epoch": "20",
        "--batch_size": "64",
        "--dim": "64",
        "--l2": "1e-7",
        "--lr": "5e-3",
        "--context_hops": "3",
        "--neighbor_samples": "8",
        "--neighbor_agg": "attention",
        "--tok_k": "4",
        "--eval_query_batch": "8",
    },
    "fb237_v1": {
        "--epoch": "20",
        "--batch_size": "64",
        "--dim": "64",
        "--l2": "1e-7",
        "--lr": "5e-3",
        "--context_hops": "2",
        "--neighbor_samples": "16",
        "--neighbor_agg": "attention",
        "--tok_k": "10",
        "--eval_query_batch": "8",
    },
    "fb237_v2": {
        "--epoch": "20",
        "--batch_size": "64",
        "--dim": "64",
        "--l2": "1e-7",
        "--lr": "5e-3",
        "--context_hops": "2",
        "--neighbor_samples": "16",
        "--neighbor_agg": "attention",
        "--tok_k": "10",
        "--eval_query_batch": "8",
    },
    "fb237_v3": {
        "--epoch": "20",
        "--batch_size": "64",
        "--dim": "64",
        "--l2": "1e-7",
        "--lr": "5e-3",
        "--context_hops": "2",
        "--neighbor_samples": "16",
        "--neighbor_agg": "attention",
        "--tok_k": "10",
        "--eval_query_batch": "8",
    },
    "fb237_v4": {
        "--epoch": "20",
        "--batch_size": "64",
        "--dim": "64",
        "--l2": "1e-7",
        "--lr": "5e-3",
        "--context_hops": "2",
        "--neighbor_samples": "16",
        "--neighbor_agg": "attention",
        "--tok_k": "10",
        "--eval_query_batch": "8",
    },
    "nell_v1": {
        "--epoch": "20",
        "--batch_size": "64",
        "--dim": "64",
        "--l2": "1e-7",
        "--lr": "5e-3",
        "--context_hops": "2",
        "--neighbor_samples": "8",
        "--neighbor_agg": "attention",
        "--tok_k": "3",
        "--eval_query_batch": "8",
    },
    "nell_v2": {
        "--epoch": "20",
        "--batch_size": "64",
        "--dim": "64",
        "--l2": "1e-7",
        "--lr": "5e-3",
        "--context_hops": "2",
        "--neighbor_samples": "8",
        "--neighbor_agg": "attention",
        "--tok_k": "3",
        "--eval_query_batch": "8",
    },
    "nell_v3": {
        "--epoch": "20",
        "--batch_size": "64",
        "--dim": "64",
        "--l2": "1e-7",
        "--lr": "5e-3",
        "--context_hops": "2",
        "--neighbor_samples": "8",
        "--neighbor_agg": "attention",
        "--tok_k": "3",
        "--eval_query_batch": "8",
    },
    "nell_v4": {
        "--epoch": "20",
        "--batch_size": "64",
        "--dim": "64",
        "--l2": "1e-7",
        "--lr": "5e-3",
        "--context_hops": "2",
        "--neighbor_samples": "8",
        "--neighbor_agg": "attention",
        "--tok_k": "3",
        "--eval_query_batch": "8",
    },
}


for dataset_name in DATASET_CONFIGS:
    final_config = COMMON_CONFIG.copy()
    final_config.update(DATASET_CONFIGS[dataset_name])
    DATASET_CONFIGS[dataset_name] = final_config


DATASET_ORDER = ["nell_v1", "nell_v2", "nell_v3", "nell_v4"]


def run_training_command(dataset_name, config):
    if not os.path.exists("main.py"):
        print("ERROR: 'main.py' not found. Run this script from Inductive/src.")
        return False

    command = list(BASE_COMMAND)
    command.extend(["--dataset", dataset_name])
    for arg, value in config.items():
        command.extend([arg, str(value)])
    if USE_CUDA_FOR_ALL:
        command.append("--cuda")

    print(f"\n{'=' * 20} Starting inductive entity prediction for: {dataset_name} {'=' * 20}")
    print(f"Executing command: {shlex.join(command)}")
    try:
        subprocess.run(command, check=True, text=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Training failed for dataset {dataset_name} with exit code {exc.returncode}")
        return False


def main_runner():
    success_count = 0
    fail_count = 0
    for dataset in DATASET_ORDER:
        if run_training_command(dataset, DATASET_CONFIGS[dataset]):
            success_count += 1
        else:
            fail_count += 1
            break

    print("-" * 50)
    print(f"Successful runs: {success_count}")
    print(f"Failed runs: {fail_count}")
    if fail_count:
        sys.exit(1)


if __name__ == "__main__":
    main_runner()
