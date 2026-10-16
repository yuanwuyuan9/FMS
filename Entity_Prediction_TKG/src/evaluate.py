import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from data_loader import default_data_directory, load_data
from model import FMS
from train import evaluate_tail, _format_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate a temporal FMS checkpoint')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data_dir', type=Path)
    parser.add_argument('--split', choices=['valid', 'test'], default='test')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--eval_batch_size', type=int)
    parser.add_argument('--eval_samples', type=int)
    options = parser.parse_args()
    checkpoint = torch.load(options.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    if (config.get('sampling_strategy', 'recency') != 'recency'
            or not config.get('history_copy', True)
            or 'history_gate_weight' not in checkpoint['state_dict']):
        raise ValueError('Checkpoint does not match the current temporal FMS architecture.')
    args = SimpleNamespace(**checkpoint['config'])
    args.cuda = not options.cpu
    if args.cuda and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --cpu.')
    args.data_dir = str(options.data_dir or default_data_directory(args.dataset))
    for name in ('eval_batch_size', 'eval_samples'):
        value = getattr(options, name)
        if value is not None:
            if value < 1:
                parser.error(f'{name} must be positive.')
            setattr(args, name, value)
    data = load_data(args)
    if data.metadata()['split_sha256'] != checkpoint['metadata']['split_sha256']:
        raise ValueError('Dataset split hashes differ from the saved training experiment.')
    model = FMS(args, data.n_relations, data.graphs[options.split], data.n_entities)
    model.load_state_dict(checkpoint['state_dict'])
    model.to('cuda' if args.cuda else 'cpu')
    if args.max_eval_queries:
        print(f'SMOKE TEST checkpoint: evaluating at most {args.max_eval_queries} queries.')
    metrics = evaluate_tail(model, getattr(data, options.split), data, args, options.split)
    print(f'[{options.split.upper()}] ' + _format_metrics(metrics['tail']))


if __name__ == '__main__':
    main()
