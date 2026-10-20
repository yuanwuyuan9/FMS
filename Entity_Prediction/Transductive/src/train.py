import json
from pathlib import Path

import numpy as np
import torch

from data_loader import load_data
from model import FMS


def train(args, data=None, evaluate_test=True):
    """Select by mean head/tail validation MRR; evaluate test once after restoring the best model."""
    output_path = Path(args.results_json) if args.results_json else None
    checkpoint_path = Path(args.checkpoint) if getattr(args, 'checkpoint', None) else None
    paths = [p.resolve() for p in (output_path, checkpoint_path) if p is not None]
    if len(set(paths)) != len(paths):
        raise ValueError('Results and checkpoint must have different paths.')
    for path in paths:
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite experiment output: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
    if data is None:
        data = load_data(args)
    device = torch.device('cuda' if args.cuda else 'cpu')
    targets, graph, stats = data.epoch_data(0)
    model = FMS(args, data.n_relations, graph, data.n_entities).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = (torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)
                 if args.decay_rate != 1.0 else None)
    metadata = data.metadata()
    metadata.update(config=vars(args).copy(), device=str(device),
                    selection_metric='validation.average.mrr',
                    parameters=sum(p.numel() for p in model.parameters()),
                    test_evaluated_during_training=False, test_evaluated_after_selection=evaluate_test,
                    model='FMS', flow_enabled=True, semantic_weighting=model.semantic_weighting,
                    classification_loss=model.classification_loss,
                    endpoint_weight=model.endpoint_weight,
                    feature_normalization=model.feature_normalization,
                    training_answer_filter='original facts + train and inverses, same (head, relation)',
                    loss='mean training-positive-filtered CE + lamda * mean CFM MSE + endpoint_weight * mean endpoint MSE')
    print('Total trainable parameters: %d' % metadata['parameters'])
    print('start training ...', flush=True)
    best = None
    best_state = None
    history = []
    cumulative_optimizer_steps = 0
    for epoch in range(args.epoch):
        if epoch > 0:
            targets, graph, stats = data.epoch_data(epoch)
        model.set_graph(graph)
        indices = np.random.permutation(len(targets))
        loss_sums = dict.fromkeys(('total', 'classification', 'cfm', 'endpoint',
                                   'weighted_cfm', 'weighted_endpoint'), 0.0)
        optimizer_steps = 0
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start:start + args.batch_size]
            # Batch indices refer to targets, not to edges in the context graph.
            batch = get_feed_dict(targets, batch_indices, device, train_answers=data.train_answers)
            losses = model.train_step(model, optimizer, batch, return_components=True)
            for name, value in losses.items():
                loss_sums[name] += value * len(batch_indices)
            optimizer_steps += 1
        cumulative_optimizer_steps += optimizer_steps
        if scheduler is not None:
            scheduler.step()

        model.set_graph(data.eval_graph)
        valid = evaluate_head_tail(model, data.valid, data, args.eval_batch_size)
        losses = {name: value / len(targets) for name, value in loss_sums.items()}
        result = dict(epoch=epoch + 1, loss=losses['total'], losses=losses, graph=stats, valid=valid, test=None,
                      optimizer_steps=optimizer_steps, cumulative_optimizer_steps=cumulative_optimizer_steps)
        history.append(result)
        print(f'epoch {epoch + 1:2d} loss: {result["loss"]:.6f} '
              f'cls: {losses["classification"]:.6f} cfm: {losses["cfm"]:.6f} '
              f'end: {losses["endpoint"]:.6f} '
              f'weighted_cfm: {losses["weighted_cfm"]:.6f} '
              f'weighted_end: {losses["weighted_endpoint"]:.6f} '
              f'facts: {stats["fact_triples"]} supervised: {len(targets)}', flush=True)
        print_metrics('VALID', valid)
        if best is None or valid['average']['mrr'] > best['valid']['average']['mrr']:
            best = result
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    # Every saved state has the same fixed evaluation graph, despite varying train graphs.
    model.load_state_dict(best_state)
    model.eval()
    best = dict(best)
    if evaluate_test:
        best['test'] = evaluate_head_tail(model, data.test, data, args.eval_batch_size)
    print(f'final results (best validation mean head/tail MRR, epoch {best["epoch"]})', flush=True)
    print_metrics('VALID', best['valid'])
    if best['test'] is not None:
        print_metrics('TEST', best['test'])
    if checkpoint_path is not None:
        with checkpoint_path.open('xb') as handle:
            torch.save(dict(config=vars(args).copy(), metadata=metadata,
                            state_dict=model.state_dict()), handle)
    if output_path is not None:
        with output_path.open('x') as handle:
            json.dump(dict(metadata=metadata, best=best, history=history), handle, indent=2)
    return model, dict(metadata=metadata, best=best, history=history)


def get_feed_dict(triplets, indices, device, training=False, train_answers=None):
    # In-memory triples use (head, tail, relation).
    rows = triplets[indices]
    triples = torch.as_tensor(rows, dtype=torch.long, device=device)
    batch = {'heads': triples[:, 0], 'relations': triples[:, 2], 'labels': triples[:, 1]}
    if training:
        batch['train_edges'] = torch.as_tensor(indices, dtype=torch.long, device=device)
    other_rows, other_tails = [], []
    if train_answers is not None:
        for row, (head, tail, relation) in enumerate(rows):
            answers = sorted(train_answers.get((int(head), int(relation)), set()) - {int(tail)})
            other_rows.extend([row] * len(answers))
            other_tails.extend(answers)
        batch['other_answers'] = (torch.tensor(other_rows, dtype=torch.long, device=device),
                                  torch.tensor(other_tails, dtype=torch.long, device=device))
    return batch


def filtered_ranks(scores, heads, relations, tails, true_ent):
    """Filter other known answers and use average ranks for tied scores."""
    scores = scores.clone()
    for i, (head, relation, tail) in enumerate(zip(heads.tolist(), relations.tolist(), tails.tolist())):
        other_answers = sorted(true_ent.get((head, relation), set()) - {tail})
        if other_answers:
            scores[i, other_answers] = -torch.inf
    true_scores = scores.gather(1, tails[:, None])
    greater = (scores > true_scores).sum(dim=1)
    equal = (scores == true_scores).sum(dim=1)
    return 1.0 + greater + (equal - 1) * 0.5


def evaluate_and_rank(model, triplets, true_ent, batch_size, neighbor_tables=None):
    if not len(triplets):
        raise ValueError("Cannot evaluate an empty split.")
    device = next(model.parameters()).device
    total_correct = 0
    all_ranks = []
    model.eval()
    original_neighbors = model.entity2edges if neighbor_tables is not None else None
    if neighbor_tables is not None:
        if not len(neighbor_tables):
            raise ValueError('At least one evaluation neighbor table is required.')
        # Sampled graphs share edge IDs. Keep their neighbor tables on the model device.
        neighbor_tables = [torch.as_tensor(table, dtype=torch.long, device=device)
                           for table in neighbor_tables]
    try:
        with torch.no_grad():
            for start in range(0, len(triplets), batch_size):
                indices = np.arange(start, min(start + batch_size, len(triplets)))
                batch = get_feed_dict(triplets, indices, device)
                if neighbor_tables is None:
                    correct, scores = model.test_step(model, batch)
                else:
                    scores = None
                    for table in neighbor_tables:
                        model.entity2edges = table
                        sampled_scores = model(batch['heads'], batch['relations'])
                        if not torch.isfinite(sampled_scores).all():
                            raise FloatingPointError('Non-finite FMS evaluation scores.')
                        if scores is None:
                            scores = sampled_scores
                        else:
                            scores = scores + sampled_scores
                    scores = scores / len(neighbor_tables)
                    correct = scores.argmax(dim=1) == batch['labels']
                if not torch.isfinite(scores).all():
                    raise FloatingPointError("Non-finite FMS evaluation scores.")
                total_correct += correct.sum().item()
                all_ranks.append(filtered_ranks(
                    scores, batch['heads'], batch['relations'], batch['labels'], true_ent).cpu())
    finally:
        if neighbor_tables is not None:
            model.entity2edges = original_neighbors
    ranks = torch.cat(all_ranks).double()
    return {
        'count': len(ranks),
        'accuracy': total_correct / len(ranks),
        'mrr': ranks.reciprocal().mean().item(),
        'mr': ranks.mean().item(),
        'h1': (ranks <= 1).double().mean().item(),
        'h3': (ranks <= 3).double().mean().item(),
        'h10': (ranks <= 10).double().mean().item(),
    }


def _format_metrics(metrics):
    return ('n: {count} acc: {accuracy:.4f} mrr: {mrr:.4f} mr: {mr:.4f} '
            'h1: {h1:.4f} h3: {h3:.4f} h10: {h10:.4f}').format(**metrics)


def evaluate_tail(model, triples, data, batch_size):
    tables = data.eval_neighbors if len(data.eval_neighbors) > 1 else None
    return dict(tail=evaluate_and_rank(model, triples, data.filters, batch_size, neighbor_tables=tables))


def evaluate_head_tail(model, triples, data, batch_size):
    tail = evaluate_tail(model, triples, data, batch_size)['tail']
    head = evaluate_tail(model, data.inverse(triples), data, batch_size)['tail']
    # Each original triple contributes one head rank and one tail rank.
    average = {name: (head[name] + tail[name]) / 2 for name in tail if name != 'count'}
    average['count'] = head['count'] + tail['count']
    return dict(head=head, tail=tail, average=average)


def print_metrics(split, metrics):
    for direction in ('head', 'tail', 'average'):
        print(f'[{split} {direction.upper()}] ' + _format_metrics(metrics[direction]), flush=True)
