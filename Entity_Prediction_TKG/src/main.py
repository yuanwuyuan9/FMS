import argparse
import math
from pathlib import Path
import random

import numpy as np
import torch

from data_loader import load_data, default_data_directory
from train import train
from run import DATASET_CONFIGS


def print_setting(args):
    print()
    print('=============================================')
    for name in ('dataset', 'epoch', 'batch_size', 'eval_batch_size', 'dim', 'l2', 'lr',
                 'decay_rate', 'seed', 'context_hops', 'neighbor_samples', 'neighbor_agg',
                 'tok_k', 'num_heads', 'eval_samples', 'feature_normalization'):
        print('%s: %s' % (name, getattr(args, name)))
    print('=============================================')
    print('           CFM Module Settings               ')
    print('=============================================')
    print('CFM Variant: %s' % args.cfm_variant)
    print('CFM MLP Dimension: %d' % args.cfm_model_dim)
    print('CFM Sigma: %s' % args.cfm_sigma)
    print('Flow steps: %d' % args.flow_steps)
    print('CFM weight: %s' % args.lamda)
    print('Endpoint weight: %s' % args.endpoint_weight)
    print()


def main():
    parser = argparse.ArgumentParser(description='Temporal FMS tail prediction (extrapolation)')
    parser.add_argument('--dataset', required=True, choices=list(DATASET_CONFIGS))
    parser.add_argument('--data_dir', type=Path, help='directory containing stat.txt and numeric train/valid/test events')
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument('--cuda', dest='cuda', action='store_true', help='require CUDA (default)')
    devices.add_argument('--cpu', dest='cuda', action='store_false', help='run locally on CPU')
    parser.set_defaults(cuda=True, use_context=True, feature_type='id', classification_loss='ce',
                        feature_normalization='none')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--eval_batch_size', type=int)
    parser.add_argument('--eval_samples', type=int, default=4,
                        help='fixed neighbor samples to average during validation/test; use 4 for fusion')
    parser.add_argument('--dim', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--l2', type=float, help='Adam weight decay')
    parser.add_argument('--decay_rate', type=float)

    parser.add_argument('--context_hops', type=int)
    parser.add_argument('--neighbor_samples', type=int)
    parser.add_argument('--neighbor_agg', choices=['mean', 'concat', 'cross', 'attention'], default='attention')
    parser.add_argument('--tok_k', type=int, help='semantic top-k; 0 retains all sampled neighbors')
    parser.add_argument('--num_heads', type=int, default=4)

    parser.add_argument('--lamda', type=float, help='CFM loss weight; 0 disables CFM supervision')
    parser.add_argument('--endpoint_weight', type=float, default=0.1,
                        help='detached tail-context endpoint MSE weight; 0 disables it')
    parser.add_argument('--cfm_model_dim', type=int)
    parser.add_argument('--cfm_variant', choices=['CFM', 'TargetCFM', 'VPCFM'], default='CFM')
    parser.add_argument('--cfm_sigma', type=float, default=0.1)
    parser.add_argument('--flow_steps', type=int, default=1)
    parser.add_argument('--no_flow', action='store_true', help='flow ablation only')

    parser.add_argument('--results_json', type=Path, help='new results file; existing files are rejected')
    parser.add_argument('--checkpoint', type=Path, help='save best model and configuration; no resume support')
    parser.add_argument('--history_mode', choices=['rolling', 'train_only'], default='rolling',
                        help='rolling observes earlier held-out snapshots; train_only freezes observed events')
    parser.add_argument('--history_window', type=int, default=120,
                        help='history length in time units; 0 keeps all strictly earlier history')
    parser.add_argument('--time_unit', type=int, help='raw timestamp units per time step; inferred from train only')
    parser.add_argument('--sampling_time_scale', type=float, help='recency sampling decay scale in time units')
    parser.add_argument('--history_time_scale', type=float, help='copy branch decay scale in time units')
    parser.add_argument('--max_train_batches', type=int, default=0, help='smoke test only; 0 trains the full split')
    parser.add_argument('--max_eval_queries', type=int, default=0, help='smoke test only; 0 evaluates the full split')
    parser.add_argument('--skip_test', action='store_true', help='validation-only parameter search')
    args = parser.parse_args()
    for name, value in DATASET_CONFIGS[args.dataset].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if min(args.history_window, args.max_train_batches, args.max_eval_queries, args.seed) < 0:
        parser.error('history window, limits and seed must be nonnegative.')
    if args.time_unit is not None and args.time_unit < 1:
        parser.error('time_unit must be positive.')
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.sampling_time_scale, args.history_time_scale)):
        parser.error('temporal decay scales must be finite and positive.')
    if any(not math.isfinite(value) for value in (args.lr, args.l2, args.decay_rate, args.lamda, args.endpoint_weight, args.cfm_sigma)):
        parser.error('all numeric hyperparameters must be finite.')
    args.data_dir = str((args.data_dir or default_data_directory(args.dataset)).resolve())
    args.results_json = str(args.results_json.resolve()) if args.results_json else None
    args.checkpoint = str(args.checkpoint.resolve()) if args.checkpoint else None
    if min(args.epoch, args.batch_size, args.eval_batch_size, args.dim, args.cfm_model_dim,
           args.context_hops, args.neighbor_samples, args.num_heads, args.flow_steps, args.eval_samples) < 1:
        parser.error('epochs, batch sizes, dimensions, hops, samples, heads and flow steps must be positive.')
    if not 0 < args.decay_rate <= 1 or args.l2 < 0 or args.lr <= 0 or args.lamda < 0 or args.endpoint_weight < 0:
        parser.error('require 0 < decay_rate <= 1, lr > 0, l2 >= 0, lamda >= 0 and endpoint_weight >= 0.')
    if args.cfm_sigma < 0 or (args.cfm_variant == 'TargetCFM' and args.cfm_sigma >= 1):
        parser.error('cfm_sigma must be nonnegative and below 1 for TargetCFM.')
    if args.cuda and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Use --cpu for an intentional CPU run.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
    print('Using CUDA' if args.cuda else 'Using CPU')
    print_setting(args)
    print(f'Extrapolation: history_mode={args.history_mode}, history_window={args.history_window}')
    print(f'Temporal sampling scale: {args.sampling_time_scale}; '
          f'historical tail scale: {args.history_time_scale}')
    if args.max_train_batches or args.max_eval_queries:
        print('SMOKE TEST: partial training/evaluation; metrics are not full benchmark results.', flush=True)
    data = load_data(args)
    train(args, data, evaluate_test=not args.skip_test)


if __name__ == '__main__':
    main()
