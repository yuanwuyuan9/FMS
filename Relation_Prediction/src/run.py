import subprocess
import sys
import shlex
import os
from datetime import datetime

BASE_COMMAND = [sys.executable, 'main.py']
USE_CUDA_FOR_ALL = True

COMMON_CFM_CONFIG = {
    '--cfm_model_dim': '128',
    '--cfm_variant': 'OTCFM',
    '--cfm_sigma': '0.1',
    '--cfm_ot_method': 'exact',
}

DATASET_CONFIGS = {
    'FB15k': {
        '--epoch': '20',
        '--batch_size': '128',
        '--dim': '64',
        '--l2': '1e-7',
        '--lr': '5e-3',
        '--feature_type': 'id',
        'flags': ['--use_context'],
        '--context_hops': '2',
        '--neighbor_samples': '16',
        '--neighbor_agg': 'attention',
        '--tok_k': '10',
        '--num_heads': '4',
    },
    'FB15k-237': {
        '--epoch': '20',
        '--batch_size': '128',
        '--dim': '64',
        '--l2': '1e-7',
        '--lr': '5e-3',
        '--feature_type': 'id',
        'flags': ['--use_context'],
        '--context_hops': '2',
        '--neighbor_samples': '16',
        '--neighbor_agg': 'attention',
        '--tok_k': '10',
        '--num_heads': '4',
    },
    'wn18': {
        '--epoch': '30',
        '--batch_size': '128',
        '--dim': '64',
        '--l2': '1e-7',
        '--lr': '5e-4',
        '--feature_type': 'id',
        'flags': ['--use_context'],
        '--context_hops': '3',
        '--neighbor_samples': '8',
        '--neighbor_agg': 'attention',
        '--tok_k': '4',
        '--num_heads': '4',
    },
    'wn18rr': {
        '--epoch': '30',
        '--batch_size': '128',
        '--dim': '64',
        '--l2': '1e-7',
        '--lr': '5e-3',
        '--feature_type': 'id',
        'flags': ['--use_context'],
        '--context_hops': '3',
        '--neighbor_samples': '8',
        '--neighbor_agg': 'attention',
        '--tok_k': '4',
        '--num_heads': '4',
    },
    'NELL995': {
        '--epoch': '30',
        '--batch_size': '128',
        '--dim': '64',
        '--l2': '1e-7',
        '--lr': '5e-4',
        '--feature_type': 'id',
        'flags': ['--use_context'],
        '--context_hops': '2',
        '--neighbor_samples': '8',
        '--neighbor_agg': 'attention',
        '--tok_k': '3',
        '--num_heads': '4',
    },
    'DDB14': {
        '--epoch': '30',
        '--batch_size': '64',
        '--dim': '64',
        '--l2': '1e-7',
        '--lr': '5e-3',
        '--feature_type': 'id',
        'flags': ['--use_context'],
        '--context_hops': '2',
        '--neighbor_samples': '8',
        '--neighbor_agg': 'attention',
        '--tok_k': '3',
        '--num_heads': '4',
    },
    'fb237_v1_ind': {
        '--epoch': '20',
        '--batch_size': '128',
        '--dim': '64',
        '--l2': '1e-7',
        '--lr': '5e-3',
        '--feature_type': 'id',
        'flags': ['--use_context'],
        '--context_hops': '2',
        '--neighbor_samples': '16',
        '--neighbor_agg': 'attention',
        '--tok_k': '10',
        '--num_heads': '4',
    },
    'fb237_v2_ind': {
        '--epoch': '20',
        '--batch_size': '128',
        '--dim': '64',
        '--l2': '1e-7',
        '--lr': '5e-3',
        '--feature_type': 'id',
        'flags': ['--use_context'],
        '--context_hops': '2',
        '--neighbor_samples': '16',
        '--neighbor_agg': 'attention',
        '--tok_k': '10',
        '--num_heads': '4',
    },
    'fb237_v3_ind': {
        '--epoch': '20',
        '--batch_size': '128',
        '--dim': '64',
        '--l2': '1e-7',
        '--lr': '5e-3',
        '--feature_type': 'id',
        'flags': ['--use_context'],
        '--context_hops': '2',
        '--neighbor_samples': '16',
        '--neighbor_agg': 'attention',
        '--tok_k': '10',
        '--num_heads': '4',
    },
    'fb237_v4_ind': {
        '--epoch': '20',
        '--batch_size': '128',
        '--dim': '64',
        '--l2': '1e-7',
        '--lr': '5e-3',
        '--feature_type': 'id',
        'flags': ['--use_context'],
        '--context_hops': '2',
        '--neighbor_samples': '16',
        '--neighbor_agg': 'attention',
        '--tok_k': '10',
        '--num_heads': '4',
    },
    'WN18RR_v1_ind': {
        '--epoch': '20',
        '--batch_size': '128',
        '--dim': '64',
        '--l2': '1e-7',
        '--lr': '5e-3',
        '--feature_type': 'id',
        'flags': ['--use_context'],
        '--context_hops': '3',
        '--neighbor_samples': '8',
        '--neighbor_agg': 'attention',
        '--tok_k': '4',
        '--num_heads': '4',
    },
    'WN18RR_v2_ind': {
        '--epoch': '20',
        '--batch_size': '128',
        '--dim': '64',
        '--l2': '1e-7',
        '--lr': '5e-3',
        '--feature_type': 'id',
        'flags': ['--use_context'],
        '--context_hops': '3',
        '--neighbor_samples': '8',
        '--neighbor_agg': 'attention',
        '--tok_k': '4',
        '--num_heads': '4',
    },
    'WN18RR_v3_ind': {
        '--epoch': '20',
        '--batch_size': '128',
        '--dim': '64',
        '--l2': '1e-7',
        '--lr': '5e-3',
        '--feature_type': 'id',
        'flags': ['--use_context'],
        '--context_hops': '3',
        '--neighbor_samples': '8',
        '--neighbor_agg': 'attention',
        '--tok_k': '4',
        '--num_heads': '4',
    },
    'WN18RR_v4_ind': {
        '--epoch': '20',
        '--batch_size': '128',
        '--dim': '64',
        '--l2': '1e-7',
        '--lr': '5e-3',
        '--feature_type': 'id',
        'flags': ['--use_context'],
        '--context_hops': '3',
        '--neighbor_samples': '8',
        '--neighbor_agg': 'attention',
        '--tok_k': '4',
        '--num_heads': '4',
    },
    'nell_v1_ind': {
        '--epoch': '20',
        '--batch_size': '128',
        '--dim': '64',
        '--l2': '1e-7',
        '--lr': '5e-3',
        '--feature_type': 'id',
        'flags': ['--use_context'],
        '--context_hops': '2',
        '--neighbor_samples': '8',
        '--neighbor_agg': 'attention',
        '--tok_k': '3',
        '--num_heads': '4',
    },
    'nell_v2_ind': {
        '--epoch': '20',
        '--batch_size': '128',
        '--dim': '64',
        '--l2': '1e-7',
        '--lr': '5e-3',
        '--feature_type': 'id',
        'flags': ['--use_context'],
        '--context_hops': '2',
        '--neighbor_samples': '8',
        '--neighbor_agg': 'attention',
        '--tok_k': '3',
        '--num_heads': '4',
    },
    'nell_v3_ind': {
        '--epoch': '20',
        '--batch_size': '128',
        '--dim': '64',
        '--l2': '1e-7',
        '--lr': '5e-3',
        '--feature_type': 'id',
        'flags': ['--use_context'],
        '--context_hops': '2',
        '--neighbor_samples': '8',
        '--neighbor_agg': 'attention',
        '--tok_k': '3',
        '--num_heads': '4',
    },
    'nell_v4_ind': {
        '--epoch': '20',
        '--batch_size': '128',
        '--dim': '64',
        '--l2': '1e-7',
        '--lr': '5e-3',
        '--feature_type': 'id',
        'flags': ['--use_context'],
        '--context_hops': '2',
        '--neighbor_samples': '8',
        '--neighbor_agg': 'attention',
        '--tok_k': '3',
        '--num_heads': '4',
    }
}

for dataset_name in DATASET_CONFIGS:
    final_config = COMMON_CFM_CONFIG.copy()
    final_config.update(DATASET_CONFIGS[dataset_name])
    DATASET_CONFIGS[dataset_name] = final_config

# Transductive
DATASET_ORDER = ['FB15k-237', 'wn18rr', 'NELL995', 'DDB14', 'FB15k', 'wn18']

# Inductive
# DATASET_ORDER = ["fb237_v1_ind", 'fb237_v2_ind', 'fb237_v3_ind','fb237_v4_ind']
# DATASET_ORDER = ['nell_v2_ind', 'nell_v3_ind','nell_v4_ind']

def run_training_command(dataset_name, config):

    if not os.path.exists('main.py'):
        print("\nERROR: 'main.py' not found in the current directory.")
        print("Please ensure 'run_all.py' is in the same directory as 'main.py'.")
        sys.exit(1)

    command = list(BASE_COMMAND)
    command.append('--dataset')
    command.append(dataset_name)


    for arg, value in config.items():
        if arg != 'flags' and value is not None:
            command.append(arg)
            command.append(str(value))


    if 'flags' in config:
        command.extend(config['flags'])


    if USE_CUDA_FOR_ALL:
        if '--cuda' not in command:
             command.append('--cuda')


    print(f"\n{'='*20} Starting Training for: {dataset_name} {'='*20}")

    print(f"Executing command: {shlex.join(command)}")
    print(f"{'=' * (40 + len(dataset_name) + 2)}\n")

    try:

        process = subprocess.run(command, check=True, text=True)
        print(f"\n{'='*20} Finished Training for: {dataset_name} {'='*20}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{'!'*20} ERROR {'!'*20}")
        print(f"Training failed for dataset: {dataset_name} with exit code {e.returncode}")
        print(f"Command was: {shlex.join(command)}")
        print(f"{'!'*47}\n")
        return False
    except Exception as e:
        print(f"\n{'!'*20} UNEXPECTED ERROR {'!'*20}")
        print(f"An unexpected error occurred while running training for {dataset_name}: {e}")
        print(f"Command was: {shlex.join(command)}")
        print(f"{'!'*59}\n")
        return False


def main_runner():
    print("Starting sequential dataset training...")
    print(f"Using Python executable: {sys.executable}")
    print("-" * 50)

    success_count = 0
    fail_count = 0

    for dataset in DATASET_ORDER:
        if dataset in DATASET_CONFIGS:
            config = DATASET_CONFIGS[dataset]
            if run_training_command(dataset, config):
                success_count += 1
            else:
                fail_count += 1
                print(f"Stopping script after failure on dataset: {dataset}")
                break
        else:
            print(f"Warning: Configuration for dataset '{dataset}' not found. Skipping.")
            fail_count += 1

    print("-" * 50)
    print("Script finished.")
    print(f"Successful runs: {success_count}")
    print(f"Failed/Skipped runs: {fail_count}")
    print("-" * 50)

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main_runner()