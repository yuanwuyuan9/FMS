import hashlib
from collections import defaultdict
from functools import reduce
from math import gcd
from pathlib import Path

import numpy as np


def default_data_directory(dataset):
    return Path(__file__).resolve().parents[1] / 'data' / dataset


class TemporalGraph:
    """Outgoing, time-sorted events; inverse edges are context only."""

    def __init__(self, events, n_entities, base_n_relations, time_unit, history_window):
        inverse = events[:, [2, 1, 0, 3]].copy()
        inverse[:, 1] += base_n_relations
        events = np.concatenate((events, inverse))
        events = events[np.lexsort((events[:, 3], events[:, 0]))]
        self.events = events
        self.n_entities = n_entities
        self.null_relation = 2 * base_n_relations
        self.time_unit = time_unit
        self.history_window = history_window
        self.offsets = np.searchsorted(events[:, 0], np.arange(n_entities + 1))

    def candidates(self, entity, query_time):
        if entity == self.n_entities:
            return self.events[:0]
        events = self.events[self.offsets[entity]:self.offsets[entity + 1]]
        # All hops share the query cutoff, as in Temporal/extrapolation RED-GNN.
        end = np.searchsorted(events[:, 3], query_time, side='left')
        start = 0
        if self.history_window:
            start = np.searchsorted(events[:, 3], query_time - self.history_window * self.time_unit)
        return events[start:end]

    def historical_tail_distribution(self, heads, relations, query_times, time_scale):
        """Recency-weighted tails of the same (head, relation), using strict history."""
        if not np.isfinite(time_scale) or time_scale <= 0:
            raise ValueError('history_time_scale must be finite and positive.')
        probabilities = np.zeros((len(heads), self.n_entities), dtype=np.float32)
        for row, (head, relation, query_time) in enumerate(zip(heads, relations, query_times)):
            events = self.candidates(head, query_time)
            events = events[events[:, 1] == relation]
            if len(events):
                # Subtracting the newest time gives the same normalized weights
                # as exp(-(query_time-edge_time)/scale), without underflowing all weights.
                age = (events[-1, 3] - events[:, 3]) / self.time_unit
                weights = np.exp(-age / time_scale)
                counts = np.bincount(events[:, 2], weights=weights, minlength=self.n_entities)
                probabilities[row] = counts / counts.sum()
        return probabilities

    def sample_tree(self, entities, query_times, hops, samples, seed, sampling_time_scale=30.0):
        """Return fixed-size trees, deterministic per (seed, node, query, hop).

        Padding propagates to padding, so it cannot expose real edges next hop.
        The query answer is never an input to the prediction tree.
        """
        if not np.isfinite(sampling_time_scale) or sampling_time_scale <= 0:
            raise ValueError('sampling_time_scale must be finite and positive.')
        entities = np.asarray(entities, dtype=np.int64).reshape(-1, 1)
        query_times = np.asarray(query_times, dtype=np.int64)
        relation_ids, deltas, masks = [], [], []
        cache = {}
        for hop in range(hops):
            children = np.full((*entities.shape, samples), self.n_entities, dtype=np.int64)
            relations = np.full(children.shape, self.null_relation, dtype=np.int64)
            times = np.zeros(children.shape, dtype=np.float32)
            valid = np.zeros(children.shape, dtype=bool)
            for row, query_time in enumerate(query_times):
                for column, entity in enumerate(entities[row]):
                    key = (int(entity), int(query_time), hop)
                    if key not in cache:
                        candidates = self.candidates(entity, query_time)
                        if len(candidates):
                            # SeedSequence avoids Python's process-randomized hash.
                            rng = np.random.default_rng(np.random.SeedSequence([seed, *key]))
                            age = (candidates[-1, 3] - candidates[:, 3]) / self.time_unit
                            weights = np.exp(-age / sampling_time_scale) + 1e-12
                            probabilities = weights / weights.sum()
                            chosen = rng.choice(len(candidates), samples, replace=len(candidates) < samples,
                                                p=probabilities)
                            cache[key] = candidates[chosen]
                        else:
                            cache[key] = None
                    selected = cache[key]
                    if selected is not None:
                        children[row, column] = selected[:, 2]
                        relations[row, column] = selected[:, 1]
                        times[row, column] = (query_time - selected[:, 3]) / self.time_unit
                        valid[row, column] = True
            entities = children.reshape(len(entities), -1)
            relation_ids.append(relations.reshape(len(entities), -1))
            deltas.append(times.reshape(len(entities), -1))
            masks.append(valid.reshape(len(entities), -1))
        return relation_ids, deltas, masks


class TemporalData:
    def __init__(self, directory, history_window=120, time_unit=None, history_mode='rolling'):
        self.directory = Path(directory).resolve()
        if history_mode not in ('rolling', 'train_only'):
            raise ValueError('history_mode must be rolling or train_only.')
        if history_window < 0 or (time_unit is not None and time_unit <= 0):
            raise ValueError('Require history_window >= 0 and time_unit > 0.')
        self.history_mode = history_mode
        self.history_window = history_window
        stats = [int(value) for value in (self.directory / 'stat.txt').read_text().split()]
        self.n_entities, self.base_n_relations = stats[:2]
        self.n_relations = 2 * self.base_n_relations
        self.train, self.valid, self.test = [self.read_events(split) for split in ('train', 'valid', 'test')]
        if not (self.train[:, 3].max() < self.valid[:, 3].min()
                and self.valid[:, 3].max() < self.test[:, 3].min()):
            raise ValueError('Extrapolation requires disjoint, ordered train < valid < test timestamps.')
        differences = np.diff(np.unique(self.train[:, 3]))
        self.time_unit = int(time_unit or reduce(gcd, differences.tolist(), 1 if not len(differences) else 0))
        observed = np.concatenate((self.train, self.valid, self.test))
        self.filters = defaultdict(set)
        for head, relation, tail, timestamp in observed:
            self.filters[(int(head), int(relation), int(timestamp))].add(int(tail))
        graph_args = (self.n_entities, self.base_n_relations, self.time_unit, history_window)
        self.graphs = {'train': TemporalGraph(self.train, *graph_args)}
        if history_mode == 'rolling':
            # These graphs contain later rows, but candidates() strictly excludes t >= query t.
            self.graphs['valid'] = TemporalGraph(np.concatenate((self.train, self.valid)), *graph_args)
            self.graphs['test'] = TemporalGraph(observed, *graph_args)
        else:
            self.graphs.update(valid=self.graphs['train'], test=self.graphs['train'])

    def read_events(self, split):
        rows = []
        with (self.directory / (split + '.txt')).open() as handle:
            for line_number, line in enumerate(handle, 1):
                fields = line.split()
                if len(fields) not in (4, 5):
                    raise ValueError(f'{split}:{line_number}: expected h r t time [unused].')
                # The supplied fifth column is an unused marker, not an end time.
                row = tuple(map(int, fields[:4]))
                head, relation, tail, timestamp = row
                if not (0 <= head < self.n_entities and 0 <= tail < self.n_entities
                        and 0 <= relation < self.base_n_relations and timestamp >= 0):
                    raise ValueError(f'{split}:{line_number}: ID or time out of range.')
                rows.append(row)
        if not rows:
            raise ValueError(f'{split} is empty.')
        events = np.asarray(rows, dtype=np.int64)
        return events[np.argsort(events[:, 3], kind='stable')]

    def metadata(self):
        return dict(protocol='temporal extrapolation', prediction_direction='tail',
                    entities=self.n_entities, base_relations=self.base_n_relations,
                    model_relations=self.n_relations, history_mode=self.history_mode,
                    history_window=self.history_window, time_unit=self.time_unit,
                    history_cutoff='query_time - window <= edge_time < query_time, every hop',
                    supervision='forward events only; inverse events used only as context',
                    evaluation='time-aware filtered average-tie ranks over all entities',
                    data_directory=str(self.directory),
                    split_counts={name: len(getattr(self, name)) for name in ('train', 'valid', 'test')},
                    time_ranges={name: [int(getattr(self, name)[:, 3].min()),
                                        int(getattr(self, name)[:, 3].max())]
                                 for name in ('train', 'valid', 'test')},
                    split_sha256={name: hashlib.sha256((self.directory / (name + '.txt')).read_bytes()).hexdigest()
                                  for name in ('train', 'valid', 'test')})


def load_data(args):
    data = TemporalData(args.data_dir, args.history_window, args.time_unit, args.history_mode)
    print(f'entities: {data.n_entities} relations: {data.base_n_relations} '
          f'train: {len(data.train)} valid: {len(data.valid)} test: {len(data.test)} '
          f'time_unit: {data.time_unit}', flush=True)
    return data
