import argparse
import math
from datetime import datetime
from pathlib import Path
import shlex
import subprocess
import sys

SRC = Path(__file__).resolve().parent
# Initial settings, shared across datasets; these are not tuned temporal results.
COMMON_CONFIG = dict(epoch=30, batch_size=64, eval_batch_size=64, dim=128,
                     cfm_model_dim=128, context_hops=2, neighbor_samples=32,
                     tok_k=16, lr=0.001, decay_rate=0.995, l2=1e-5, lamda=0.1,
                     sampling_time_scale=30.0, history_time_scale=30.0)
DATASET_CONFIGS = {name: COMMON_CONFIG.copy() for name in
                   ('ICEWS14', 'ICEWS05-15', 'ICEWS18', 'GDELT', 'WIKI', 'YAGO')}


def training_command(dataset, options, output):
    config = DATASET_CONFIGS[dataset].copy()
    if options.epoch is not None:
        config['epoch'] = options.epoch
    for name in ('sampling_time_scale', 'history_time_scale'):
        if getattr(options, name) is not None:
            config[name] = getattr(options, name)
    command = [sys.executable, '-u', str(SRC / 'main.py'), '--dataset', dataset,
               '--cpu' if options.cpu else '--cuda', '--seed', str(options.seed),
               '--neighbor_agg', 'attention', '--num_heads', '4',
               '--cfm_variant', 'CFM', '--cfm_sigma', '0.1', '--flow_steps', '1',
               '--endpoint_weight', str(getattr(options, 'endpoint_weight', 0.1)),
               '--eval_samples', str(getattr(options, 'eval_samples', 4)),
               '--history_mode', options.history_mode,
               '--history_window', str(options.history_window),
               '--results_json', str(output / (dataset + '.json')),
               '--checkpoint', str(output / (dataset + '.pt'))]
    for key, value in config.items():
        command.extend(['--' + key, str(value)])
    if options.skip_test:
        command.append('--skip_test')
    return command


def run_training_command(dataset_name, command, output):
    print('\n%s Starting Training for: %s %s' % ('=' * 20, dataset_name, '=' * 20), flush=True)
    print('Executing command: %s' % shlex.join(command), flush=True)
    with (output / (dataset_name + '.log')).open('x') as log:
        log.write(shlex.join(command) + '\n')
        log.flush()
        with subprocess.Popen(command, cwd=SRC, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        if code:
            raise SystemExit(code)


def main_runner():
    parser = argparse.ArgumentParser(description='Run temporal FMS extrapolation experiments')
    parser.add_argument('--dataset', nargs='+', choices=list(DATASET_CONFIGS), default=['ICEWS14'])
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--epoch', type=int, help='override all selected epoch counts, e.g. 2 for a smoke test')
    parser.add_argument('--skip_test', action='store_true')
    parser.add_argument('--eval_samples', type=int, default=4)
    parser.add_argument('--endpoint_weight', type=float, default=0.1)
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--history_mode', choices=['rolling', 'train_only'], default='rolling')
    parser.add_argument('--history_window', type=int, default=120)
    parser.add_argument('--sampling_time_scale', type=float)
    parser.add_argument('--history_time_scale', type=float)

    options = parser.parse_args()
    if options.seed < 0:
        parser.error('--seed must be nonnegative.')
    for value in (options.sampling_time_scale, options.history_time_scale):
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error('temporal decay scales must be finite and positive.')
    if options.history_window < 0:
        parser.error('--history_window must be nonnegative.')
    if options.eval_samples < 1:
        parser.error('--eval_samples must be positive.')
    if not math.isfinite(options.endpoint_weight) or options.endpoint_weight < 0:
        parser.error('--endpoint_weight must be finite and nonnegative.')
    if options.epoch is not None and options.epoch < 1:
        parser.error('--epoch must be positive.')
    if len(set(options.dataset)) != len(options.dataset):
        parser.error('do not repeat a dataset in the same run.')
    output = (options.output_dir or SRC.parent / 'results' /
              datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for dataset in options.dataset:
        for suffix in ('.json', '.pt', '.log'):
            if (output / (dataset + suffix)).exists():
                parser.error(f'refusing to overwrite {output / (dataset + suffix)}')
    print(f'Output directory: {output}', flush=True)
    for dataset in options.dataset:
        command = training_command(dataset, options, output)
        run_training_command(dataset, command, output)


if __name__ == '__main__':
    main_runner()
