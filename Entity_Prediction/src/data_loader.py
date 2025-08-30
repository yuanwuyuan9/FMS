import os
import re
import pickle
import numpy as np
from collections import defaultdict

# A dictionary where the key is an entity ID and the value is a set of
# all edges connected to this entity (both incoming and outgoing).
entity2edge_set = defaultdict(set)

# A list where each entry represents the sampled neighboring edges for a corresponding entity.
entity2edges = []

# A list where each entry represents the two entities connected by the corresponding edge.
edge2entities = []

# A list where each entry represents the relation type of the corresponding edge.
edge2relation = []

# A dictionary where the key is an entity index and the value is a set of
# (relation, entity) pairs connected to it.
e2re = defaultdict(set)

def read_entities(file_name):
    d = {}
    file = open(file_name)
    for line in file:
        index, name = line.strip().split('\t')
        d[name] = int(index)
    file.close()

    return d


def read_relations(file_name):
    d = {}
    file = open(file_name)
    for line in file:
        index, name = line.strip().split('\t')
        d[name] = int(index)

    return d


def read_triplets(file_name):
    data = []

    file = open(file_name)
    for line in file:
        head, relation, tail = line.strip().split('\t')

        head_idx = entity_dict[head]
        relation_idx = relation_dict[relation]
        tail_idx = entity_dict[tail]

        data.append((head_idx, tail_idx, relation_idx))
    file.close()

    return data


def build_kg(train_data):
    """
    Builds the knowledge graph data structures.

    Args:
        train_data (list of tuples): The training data, where each tuple contains (head_idx, tail_idx, relation_idx).
    """
    for edge_idx, triplet in enumerate(train_data):
        head_idx, tail_idx, relation_idx = triplet

        # If using context information
        if args.use_context:
            # Map the head and tail entities to the edge index
            entity2edge_set[head_idx].add(edge_idx)
            entity2edge_set[tail_idx].add(edge_idx)
            # Record the entity indices at both ends of the edge
            edge2entities.append([head_idx, tail_idx])
            # Record the relation index of the edge
            edge2relation.append(relation_idx)

    # To handle the case where a node does not appear in the training data (i.e., this node has no neighbor edge),
    # we introduce a null entity (ID: n_entities), a null edge (ID: n_edges), and a null relation (ID: n_relations).
    # entity2edge_set[isolated_node] = {null_edge}
    # entity2edge_set[null_entity] = {null_edge}
    # edge2entities[null_edge] = [null_entity, null_entity]
    # edge2relation[null_edge] = null_relation
    # The feature of null_relation is a zero vector. See _build_model() of model.py for details

    if args.use_context:
        # Add a null entity and null relationship as placeholders for unconnected entities and relationships
        null_entity = len(entity_dict)
        null_relation = len(relation_dict)
        null_edge = len(edge2entities)
        edge2entities.append([null_entity, null_entity])
        edge2relation.append(null_relation)

        # For each entity, add sampled neighboring edges
        for i in range(len(entity_dict) + 1):
            if i not in entity2edge_set:
                entity2edge_set[i] = {null_edge}
            sampled_neighbors = np.random.choice(list(entity2edge_set[i]), size=args.neighbor_samples,
                                                 replace=len(entity2edge_set[i]) < args.neighbor_samples)
            entity2edges.append(sampled_neighbors)


entity_dict = []
relation_dict = []

def load_data(model_args):
    global args, entity_dict, relation_dict
    args = model_args
    directory = '../data/' + args.dataset + '/'

    print('reading entity dict and relation dict ...')
    entity_dict = read_entities(directory + 'entities.dict')
    relation_dict = read_relations(directory + 'relations.dict')

    print('reading train, validation, and test data ...')
    train_triplets = read_triplets(directory + 'train.txt')
    valid_triplets = read_triplets(directory + 'valid.txt')
    test_triplets = read_triplets(directory + 'test.txt')

    print('processing the knowledge graph ...')
    build_kg(train_triplets)

    triplets = [train_triplets, valid_triplets, test_triplets]

    if args.use_context:
        neighbor_params = [np.array(entity2edges), np.array(edge2entities), np.array(edge2relation)]
    else:
        neighbor_params = None


    return triplets, len(relation_dict), neighbor_params, len(entity_dict)

def load_entity_relation():
    return entity_dict, relation_dict