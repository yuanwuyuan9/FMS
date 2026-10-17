import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np


def default_data_directory(dataset):
    names = {'FB15k-237': 'FB15k-237', 'nell': 'nell', 'umls': 'umls'}
    return Path(__file__).resolve().parents[1] / 'data' / names[dataset]


def build_kg(triples, n_entities, n_relations, neighbor_samples, rng):
    # Triplets are stored as (head, tail, relation). Only sample outgoing edges.
    triples = np.asarray(triples, dtype=np.int64).reshape(-1, 3)
    null_edge = len(triples)
    outgoing = defaultdict(set)
    for edge, (head, _, _) in enumerate(triples):
        outgoing[head].add(edge)
    neighbors = []
    for entity in range(n_entities + 1):
        candidates = sorted(outgoing.get(entity, {null_edge}))
        neighbors.append(rng.choice(candidates, neighbor_samples,
                                    replace=len(candidates) < neighbor_samples))
    endpoints = np.vstack((triples[:, :2], [[n_entities, n_entities]]))
    relations = np.append(triples[:, 2], n_relations)
    return [np.asarray(neighbors), endpoints, relations]


class TransductiveData:
    def __init__(self, directory, neighbor_samples, seed, eval_samples=1):
        if eval_samples < 1:
            raise ValueError('eval_samples must be positive.')
        self.directory = Path(directory).resolve()
        self.neighbor_samples = neighbor_samples
        self.partition_rng = np.random.RandomState(seed)
        self.neighbor_rng = np.random.RandomState(seed)
        with (self.directory / 'entities.txt').open() as handle:
            self.entities = {name.strip(): i for i, name in enumerate(handle)}
        with (self.directory / 'relations.txt').open() as handle:
            self.relations = {name.strip(): i for i, name in enumerate(handle)}
        self.n_entities = len(self.entities)
        self.base_n_relations = len(self.relations)
        self.n_relations = 2 * self.base_n_relations
        self.facts = self.read_triplets('facts')
        self.train = self.read_triplets('train')
        self.valid = self.read_triplets('valid')
        self.test = self.read_triplets('test')
        self.pool = np.concatenate((self.facts, self.train))
        if not len(self.facts) or not len(self.train) or not len(self.valid) or not len(self.test):
            raise ValueError('facts, train, valid and test must all be nonempty.')
        pool_set = set(map(tuple, self.pool))
        if len(pool_set) != len(self.pool):
            raise ValueError('Duplicate facts/train triples would cross the random graph/target partition.')
        if pool_set & (set(map(tuple, self.valid)) | set(map(tuple, self.test))):
            raise ValueError('Held-out triples overlap the observed facts/train pool.')

        self.eval_graph = build_kg(self.double(self.pool), self.n_entities,
                                       self.n_relations, neighbor_samples, np.random.RandomState(seed))
        self.eval_seeds = [(seed + i) % (2 ** 32) for i in range(eval_samples)]
        self.eval_neighbors = [self.eval_graph[0]]
        for eval_seed in self.eval_seeds[1:]:
            graph = build_kg(self.double(self.pool), self.n_entities, self.n_relations,
                                 neighbor_samples, np.random.RandomState(eval_seed))
            self.eval_neighbors.append(graph[0])
        # Training answers are separate from held-out evaluation filters.
        self.train_answers = defaultdict(set)
        for head, tail, relation in self.pool:
            self.train_answers[(int(head), int(relation))].add(int(tail))
        self.filters = defaultdict(set)
        for head, tail, relation in np.concatenate((self.pool, self.valid, self.test)):
            self.filters[(int(head), int(relation))].add(int(tail))

    def read_triplets(self, split):
        triples = []
        with (self.directory / (split + '.txt')).open() as handle:
            for line in handle:
                head, relation, tail = line.split()
                triples.append((self.entities[head], self.entities[tail], self.relations[relation]))
        return np.asarray(triples, dtype=np.int64).reshape(-1, 3)

    def inverse(self, triples):
        inverse = np.asarray(triples, dtype=np.int64)[:, [1, 0, 2]].copy()
        inverse[:, 2] += self.base_n_relations
        return inverse

    def double(self, triples):
        return np.concatenate((triples, self.inverse(triples)))

    def epoch_data(self, epoch):
        if epoch == 0:
            facts, targets = self.facts, self.train
        else:
            shuffled = self.pool[self.partition_rng.permutation(len(self.pool))]
            boundary = len(shuffled) * 3 // 4
            facts, targets = shuffled[:boundary], shuffled[boundary:]
        # Add inverse edges after the split, so targets are absent in both directions.
        graph = build_kg(self.double(facts), self.n_entities, self.n_relations,
                             self.neighbor_samples, self.neighbor_rng)
        stats = dict(fact_triples=len(facts), target_triples=len(targets),
                     supervised_triples=len(targets),
                     fact_sha256=hashlib.sha256(facts.tobytes()).hexdigest(),
                     target_sha256=hashlib.sha256(targets.tobytes()).hexdigest())
        return targets, graph, stats

    def metadata(self):
        return dict(protocol='redgnn', data_directory=str(self.directory),
                    original_facts=len(self.facts), original_train=len(self.train),
                    observed_pool=len(self.pool), original_valid=len(self.valid), original_test=len(self.test),
                    entities=self.n_entities, base_relations=self.base_n_relations,
                    model_relations=self.n_relations, context_direction='outgoing',
                    prediction_direction='tail', semantic_weighting='softmax',
                    training_supervision='forward triples only',
                    graph_encoder='FMS sampled outgoing-edge context',
                    partition='original facts/train in epoch 1; then random 3:1 split before inverses',
                    evaluation_graph='fixed sampled context from original facts+train and their inverses',
                    eval_samples=len(self.eval_neighbors), eval_seeds=self.eval_seeds,
                    score_fusion='arithmetic mean of raw logits before filtering/ranking',
                    evaluation='tail-only filtered average-tie ranks',
                    rng='separate partition/training-neighbor/evaluation-neighbor RNGs with the same seed',
                    split_sha256={name: hashlib.sha256((self.directory / (name + '.txt')).read_bytes()).hexdigest()
                                  for name in ('facts', 'train', 'valid', 'test')})


def load_data(args):
    print('reading entity dict and relation dict ...')
    data = TransductiveData(args.data_dir, args.neighbor_samples, args.seed,
                            eval_samples=getattr(args, 'eval_samples', 1))
    print('n_entities: %d   n_relations: %d' % (data.n_entities, data.base_n_relations))
    print('facts: %d   train: %d   valid: %d   test: %d' %
          (len(data.facts), len(data.train), len(data.valid), len(data.test)))
    return data
