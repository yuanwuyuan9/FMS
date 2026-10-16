import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from model import FMS
from data_loader import TransductiveData, default_data_directory
from train import evaluate_tail, _format_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate FMS tail prediction')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data_dir', type=Path, help='override saved data location')
    parser.add_argument('--eval_samples', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int)
    parser.add_argument('--results_json', type=Path, required=True)
    parser.add_argument('--skip_test', action='store_true')
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument('--cuda', dest='cuda', action='store_true')
    devices.add_argument('--cpu', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)
    options = parser.parse_args()
    if options.eval_samples < 1 or (options.eval_batch_size is not None and options.eval_batch_size < 1):
        parser.error('eval_samples and eval_batch_size must be positive.')
    if options.cuda and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Use --cpu for an intentional CPU run.')
    output = options.results_json.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite experiment output: {output}')
    checkpoint = options.checkpoint.resolve()
    saved = torch.load(checkpoint, map_location='cpu')
    args = SimpleNamespace(**saved['config'])
    # Resolve data relative to the current installation.
    directory = options.data_dir or default_data_directory(args.dataset)
    data = TransductiveData(directory, args.neighbor_samples, args.seed, options.eval_samples)
    original_metadata = saved['metadata']
    if data.metadata()['split_sha256'] != original_metadata['split_sha256']:
        raise ValueError('Data split hashes differ from the checkpoint experiment.')
    device = torch.device('cuda' if options.cuda else 'cpu')
    model = FMS(args, data.n_relations, data.eval_graph, data.n_entities)
    model.load_state_dict(saved['state_dict'])
    model.to(device)
    # Do not rely on a checkpoint's sampled graph buffers after relocating/loading.
    model.set_graph(data.eval_graph)
    batch_size = options.eval_batch_size or args.eval_batch_size
    metadata = data.metadata()
    metadata.update(checkpoint=str(checkpoint), device=str(device), eval_batch_size=batch_size,
                    checkpoint_eval_samples=saved['config'].get('eval_samples', 1),
                    selection='fixed supplied checkpoint; no new model selection',
                    test_evaluated=not options.skip_test)
    print(json.dumps(metadata, indent=2), flush=True)
    valid = evaluate_tail(model, data.valid, data, batch_size)
    print('[VALID] ' + _format_metrics(valid['tail']), flush=True)
    test = None
    if not options.skip_test:
        test = evaluate_tail(model, data.test, data, batch_size)
        print('[TEST]  ' + _format_metrics(test['tail']), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(dict(metadata=metadata, valid=valid, test=test), handle, indent=2)


if __name__ == '__main__':
    main()
