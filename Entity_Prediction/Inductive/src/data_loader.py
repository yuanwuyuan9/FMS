from pathlib import Path
from collections import defaultdict
import numpy as np


PREDICT_TAIL = 0
PREDICT_HEAD = 1


def _read_mapping(path):
    mapping = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 2:
                raise ValueError(f"Invalid mapping line in {path}: {line!r}")
            key, value = parts
            try:
                mapping[key] = int(value)
            except ValueError:
                mapping[value] = int(key)
    return mapping


def _read_triplets(path, entity_dict, relation_dict):
    triplets = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 3:
                raise ValueError(f"Invalid triple line in {path}: {line!r}")
            head, relation, tail = parts
            triplets.append((
                entity_dict[head],
                entity_dict[tail],
                relation_dict[relation],
            ))
    return triplets


def _add_inverse_relations(relation_dict):
    offset = max(relation_dict.values()) + 1 if relation_dict else 0
    for relation_name, relation_idx in list(relation_dict.items()):
        relation_dict[f"{relation_name}_inverse"] = relation_idx + offset
    return offset


def _build_context_graph(triplets, n_entities, n_relations, inverse_relation_offset, neighbor_samples):
    entity2edge_set = defaultdict(set)
    edge2entities = []
    edge2relation = []

    for head_idx, tail_idx, relation_idx in triplets:
        forward_edge_idx = len(edge2entities)
        entity2edge_set[head_idx].add(forward_edge_idx)
        entity2edge_set[tail_idx].add(forward_edge_idx)
        edge2entities.append([head_idx, tail_idx])
        edge2relation.append(relation_idx)

        inverse_edge_idx = len(edge2entities)
        entity2edge_set[tail_idx].add(inverse_edge_idx)
        entity2edge_set[head_idx].add(inverse_edge_idx)
        edge2entities.append([tail_idx, head_idx])
        edge2relation.append(relation_idx + inverse_relation_offset)

    null_entity = n_entities
    null_relation = n_relations
    null_edge = len(edge2entities)
    edge2entities.append([null_entity, null_entity])
    edge2relation.append(null_relation)

    entity2edges = []
    for entity_idx in range(n_entities + 1):
        neighbors = entity2edge_set.get(entity_idx, {null_edge})
        sampled_neighbors = np.random.choice(
            list(neighbors),
            size=neighbor_samples,
            replace=len(neighbors) < neighbor_samples,
        )
        entity2edges.append(sampled_neighbors)

    return [
        np.asarray(entity2edges, dtype=np.int64),
        np.asarray(edge2entities, dtype=np.int64),
        np.asarray(edge2relation, dtype=np.int64),
    ]


def _build_directed_examples(triplets, inverse_relation_offset, edge_offset=None):
    known_entities = []
    target_entities = []
    relations = []
    target_sides = []
    source_triplets = []
    mask_edges = []

    for idx, triplet in enumerate(triplets):
        head, tail, relation = triplet
        if edge_offset is None:
            edge_mask = [-1, -1]
        else:
            forward_edge = (edge_offset + idx) * 2
            edge_mask = [forward_edge, forward_edge + 1]

        known_entities.append(head)
        target_entities.append(tail)
        relations.append(relation)
        target_sides.append(PREDICT_TAIL)
        source_triplets.append(triplet)
        mask_edges.append(edge_mask)

        known_entities.append(tail)
        target_entities.append(head)
        relations.append(relation + inverse_relation_offset)
        target_sides.append(PREDICT_HEAD)
        source_triplets.append(triplet)
        mask_edges.append(edge_mask)

    return {
        "known_entities": np.asarray(known_entities, dtype=np.int64),
        "target_entities": np.asarray(target_entities, dtype=np.int64),
        "relations": np.asarray(relations, dtype=np.int64),
        "target_sides": target_sides,
        "source_triplets": source_triplets,
        "mask_edges": np.asarray(mask_edges, dtype=np.int64),
    }


def _concat_examples(*examples):
    return {
        "known_entities": np.concatenate([example["known_entities"] for example in examples]),
        "target_entities": np.concatenate([example["target_entities"] for example in examples]),
        "relations": np.concatenate([example["relations"] for example in examples]),
        "target_sides": sum((example["target_sides"] for example in examples), []),
        "source_triplets": sum((example["source_triplets"] for example in examples), []),
        "mask_edges": np.concatenate([example["mask_edges"] for example in examples]),
    }


def _build_true_entities(*triplet_groups, inverse_relation_offset):
    true_entities = defaultdict(set)
    for triplets in triplet_groups:
        for head, tail, relation in triplets:
            true_entities[(head, relation)].add(tail)
            true_entities[(tail, relation + inverse_relation_offset)].add(head)
    return true_entities


def _entity_names(path):
    return set(_read_mapping(path).keys())


def load_data(args):
    dataset = args.dataset[:-4] if args.dataset.endswith("_ind") else args.dataset
    base_dir = Path(__file__).resolve().parents[1] / "data" / dataset
    ind_dir = Path(str(base_dir) + "_ind")

    if not base_dir.exists():
        raise FileNotFoundError(f"Base inductive dataset directory not found: {base_dir}")
    if not ind_dir.exists():
        raise FileNotFoundError(f"Inductive dataset directory not found: {ind_dir}")

    print("reading base and inductive entity/relation dictionaries ...")
    base_entity_dict = _read_mapping(base_dir / "entities.txt")
    ind_entity_dict = _read_mapping(ind_dir / "entities.txt")
    relation_dict = _read_mapping(base_dir / "relations.txt")
    inverse_relation_offset = _add_inverse_relations(relation_dict)
    args.inverse_relation_offset = inverse_relation_offset

    print("reading support/query triples ...")
    base_support = _read_triplets(base_dir / "train.txt", base_entity_dict, relation_dict)
    base_extra_train = _read_triplets(base_dir / "valid.txt", base_entity_dict, relation_dict)
    base_valid = _read_triplets(base_dir / "test.txt", base_entity_dict, relation_dict)

    ind_support = _read_triplets(ind_dir / "train.txt", ind_entity_dict, relation_dict)
    ind_valid = _read_triplets(ind_dir / "valid.txt", ind_entity_dict, relation_dict)
    ind_test = _read_triplets(ind_dir / "test.txt", ind_entity_dict, relation_dict)

    print("building base and inductive context graphs ...")
    base_graph = _build_context_graph(
        base_support,
        len(base_entity_dict),
        len(relation_dict),
        inverse_relation_offset,
        args.neighbor_samples,
    )
    ind_graph = _build_context_graph(
        ind_support,
        len(ind_entity_dict),
        len(relation_dict),
        inverse_relation_offset,
        args.neighbor_samples,
    )

    support_train_examples = _build_directed_examples(
        base_support,
        inverse_relation_offset,
        edge_offset=0,
    )
    extra_train_examples = _build_directed_examples(
        base_extra_train,
        inverse_relation_offset,
        edge_offset=None,
    )
    train_examples = _concat_examples(support_train_examples, extra_train_examples)

    base_valid_examples = _build_directed_examples(base_valid, inverse_relation_offset)
    ind_valid_examples = _build_directed_examples(ind_valid, inverse_relation_offset)
    ind_test_examples = _build_directed_examples(ind_test, inverse_relation_offset)
    ind_all_examples = _concat_examples(ind_valid_examples, ind_test_examples)

    base_true_entities = _build_true_entities(
        base_support,
        base_extra_train,
        base_valid,
        inverse_relation_offset=inverse_relation_offset,
    )
    ind_true_entities = _build_true_entities(
        ind_support,
        ind_valid,
        ind_test,
        inverse_relation_offset=inverse_relation_offset,
    )

    overlap = len(_entity_names(base_dir / "entities.txt") & _entity_names(ind_dir / "entities.txt"))
    print(
        "inductive split summary: "
        f"base_entities={len(base_entity_dict)} "
        f"ind_entities={len(ind_entity_dict)} "
        f"entity_overlap={overlap}"
    )
    print(
        "query split summary: "
        f"train_examples={len(train_examples['target_entities'])} "
        f"support_train_triples={len(base_support)} "
        f"extra_train_triples={len(base_extra_train)} "
        f"base_valid_triples={len(base_valid)} "
        f"ind_valid_triples={len(ind_valid)} "
        f"ind_test_triples={len(ind_test)}"
    )

    return {
        "dataset": dataset,
        "n_relations": len(relation_dict),
        "inverse_relation_offset": inverse_relation_offset,
        "n_entities": {
            "base": len(base_entity_dict),
            "inductive": len(ind_entity_dict),
        },
        "graphs": {
            "base": base_graph,
            "inductive": ind_graph,
        },
        "examples": {
            "train": train_examples,
            "base_valid": base_valid_examples,
            "ind_valid": ind_valid_examples,
            "ind_test": ind_test_examples,
            "ind_all": ind_all_examples,
        },
        "true_entities": {
            "base": base_true_entities,
            "inductive": ind_true_entities,
        },
    }
