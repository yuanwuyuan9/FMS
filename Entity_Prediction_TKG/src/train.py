import json
from pathlib import Path

import numpy as np
import torch

from data_loader import load_data
from model import FMS


def get_feed_dict(events, indices, device):
    events = torch.as_tensor(events[indices], dtype=torch.long, device=device)
    return dict(heads=events[:, 0], relations=events[:, 1], labels=events[:, 2], timestamps=events[:, 3])


def filtered_ranks(scores, batch, filters):
    scores = scores.clone()
    for i, (head, relation, tail, timestamp) in enumerate(zip(
            batch['heads'].tolist(), batch['relations'].tolist(),
            batch['labels'].tolist(), batch['timestamps'].tolist())):
        other_answers = sorted(filters.get((head, relation, timestamp), set()) - {tail})
        if other_answers:
            scores[i, other_answers] = -torch.inf
    true_scores = scores.gather(1, batch['labels'][:, None])
    greater = (scores > true_scores).sum(dim=1)
    equal = (scores == true_scores).sum(dim=1)
    return 1.0 + greater + (equal - 1) * 0.5


def evaluate_tail(model, events, data, args, split):
    if not len(events):
        raise ValueError('Cannot evaluate an empty split.')
    original_graph, original_seed = model.graph, model.sampling_seed
    device = next(model.parameters()).device
    ranks, correct = [], 0
    limit = min(len(events), args.max_eval_queries or len(events))
    model.eval()
    try:
        model.set_graph(data.graphs[split])
        with torch.no_grad():
            for start in range(0, limit, args.eval_batch_size):
                batch = get_feed_dict(events, np.arange(start, min(start + args.eval_batch_size, limit)), device)
                scores = None
                for sample in range(args.eval_samples):
                    model.sampling_seed = args.seed + sample
                    sampled = model(batch['heads'], batch['relations'], batch['timestamps'])
                    if not torch.isfinite(sampled).all():
                        raise FloatingPointError('Non-finite temporal FMS scores.')
                    # FMS returns log probabilities; fuse sampled predictions in probability space.
                    sampled = sampled.exp()
                    scores = sampled if scores is None else scores + sampled
                scores = scores / args.eval_samples
                correct += (scores.argmax(dim=1) == batch['labels']).sum().item()
                ranks.append(filtered_ranks(scores, batch, data.filters).cpu())
    finally:
        model.set_graph(original_graph)
        model.sampling_seed = original_seed
    ranks = torch.cat(ranks).double()
    return dict(tail=dict(count=len(ranks), accuracy=correct / len(ranks),
                          mrr=ranks.reciprocal().mean().item(), mr=ranks.mean().item(),
                          h1=(ranks <= 1).double().mean().item(),
                          h3=(ranks <= 3).double().mean().item(),
                          h10=(ranks <= 10).double().mean().item()))


def _format_metrics(metrics):
    return ('n: {count} acc: {accuracy:.4f} mrr: {mrr:.4f} mr: {mr:.4f} '
            'h1: {h1:.4f} h3: {h3:.4f} h10: {h10:.4f}').format(**metrics)


def train(args, data=None, evaluate_test=True):
    paths = [Path(path).resolve() for path in (args.results_json, args.checkpoint) if path]
    if len(set(paths)) != len(paths):
        raise ValueError('Results and checkpoint must use different paths.')
    for path in paths:
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite experiment output: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
    data = data if data is not None else load_data(args)
    device = torch.device('cuda' if args.cuda else 'cpu')
    model = FMS(args, data.n_relations, data.graphs['train'], data.n_entities).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)
    metadata = data.metadata()
    metadata.update(config=vars(args).copy(), device=str(device),
                    parameters=sum(p.numel() for p in model.parameters()),
                    selection_metric='validation.tail.mrr',
                    test_evaluated_during_training=False,
                    test_evaluated_after_selection=evaluate_test,
                    flow_enabled=model.use_flow,
                    eval_samples=args.eval_samples,
                    score_fusion='arithmetic mean of fixed sampled probabilities',
                    temporal_sampling='exponentially time-weighted outgoing events',
                    sampling_time_scale=model.sampling_time_scale,
                    temporal_prediction='learned gating of flow and historical tail distributions',
                    history_time_scale=model.history_time_scale,
                    loss='mean CE + lamda * mean CFM MSE + endpoint_weight * mean endpoint MSE',
                    smoke_test=bool(args.max_train_batches or args.max_eval_queries))
    print('Total trainable parameters: %d' % metadata['parameters'], flush=True)
    history, best, best_state = [], None, None
    for epoch in range(args.epoch):
        model.set_graph(data.graphs['train'])
        indices = np.random.permutation(len(data.train))
        if args.max_train_batches:
            indices = indices[:args.max_train_batches * args.batch_size]
        sums = dict.fromkeys(('total', 'classification', 'cfm', 'endpoint', 'weighted_cfm', 'weighted_endpoint'), 0.0)
        for start in range(0, len(indices), args.batch_size):
            selected = indices[start:start + args.batch_size]
            model.sampling_seed = int(np.random.randint(0, 2 ** 32))
            batch = get_feed_dict(data.train, selected, device)
            losses = model.train_step(model, optimizer, batch, return_components=True)
            for name, value in losses.items():
                sums[name] += len(selected) * value
            if (start // args.batch_size + 1) % 100 == 0:
                print(f'epoch {epoch + 1} trained {min(start + args.batch_size, len(indices))}/{len(indices)}', flush=True)
        scheduler.step()
        losses = {name: value / len(indices) for name, value in sums.items()}
        valid = evaluate_tail(model, data.valid, data, args, 'valid')
        result = dict(epoch=epoch + 1, loss=losses['total'], losses=losses, valid=valid,
                      test=None, trained_queries=len(indices))
        history.append(result)
        print(f'epoch {epoch + 1} loss: {losses["total"]:.6f} cls: {losses["classification"]:.6f} '
              f'cfm: {losses["cfm"]:.6f} end: {losses["endpoint"]:.6f}', flush=True)
        print('[VALID] ' + _format_metrics(valid['tail']), flush=True)
        if best is None or valid['tail']['mrr'] > best['valid']['tail']['mrr']:
            best = result
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    best = dict(best)
    if evaluate_test:
        best['test'] = evaluate_tail(model, data.test, data, args, 'test')
        print('[TEST] ' + _format_metrics(best['test']['tail']), flush=True)
    print(f'Best validation epoch: {best["epoch"]}', flush=True)
    report = dict(metadata=metadata, best=best, history=history)
    if args.checkpoint:
        with Path(args.checkpoint).open('xb') as handle:
            torch.save(dict(config=vars(args).copy(), metadata=metadata, state_dict=best_state), handle)
    if args.results_json:
        with Path(args.results_json).open('x') as handle:
            json.dump(report, handle, indent=2)
    return model, report
