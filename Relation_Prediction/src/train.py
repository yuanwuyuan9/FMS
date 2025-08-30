import torch
import numpy as np
from collections import defaultdict
from model import FMS
from datetime import datetime

args = None


def train(model_args, data):

    global args, model, sess
    args = model_args


    triplets, n_relations, neighbor_params, n_entity = data
    train_triplets, valid_triplets, test_triplets = triplets


    train_edges = torch.LongTensor(np.array(range(len(train_triplets)), np.int32))
    train_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in train_triplets], np.int32))
    valid_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in valid_triplets], np.int32))
    test_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in test_triplets], np.int32))

    train_labels = torch.LongTensor(np.array([triplet[2] for triplet in train_triplets], np.int32))
    valid_labels = torch.LongTensor(np.array([triplet[2] for triplet in valid_triplets], np.int32))
    test_labels = torch.LongTensor(np.array([triplet[2] for triplet in test_triplets], np.int32))

    model = FMS(args, n_relations, neighbor_params, n_entity)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.l2,
    )

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}, Requires Grad: {param.requires_grad}")
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.cuda:
        model = model.cuda()
        train_labels = train_labels.cuda()
        valid_labels = valid_labels.cuda()
        test_labels = test_labels.cuda()
        if args.use_context:
            train_edges = train_edges.cuda()
            train_entity_pairs = train_entity_pairs.cuda()
            valid_entity_pairs = valid_entity_pairs.cuda()
            test_entity_pairs = test_entity_pairs.cuda()


    true_relations = defaultdict(set)
    for head, tail, relation in train_triplets + valid_triplets + test_triplets:
        true_relations[(head, tail)].add(relation)

    best_valid_acc = 0.0
    final_res = None

    print('start training ...')
    for step in range(args.epoch):

        index = np.arange(len(train_labels))
        np.random.shuffle(index)
        if args.use_context:
            train_entity_pairs = train_entity_pairs[index]
            train_edges = train_edges[index]
        train_labels = train_labels[index]
        s = 0
        while s + args.batch_size <= len(train_labels):

            loss = model.train_step(model, optimizer, get_feed_dict(
                train_entity_pairs, train_edges, train_labels, s, s + args.batch_size))
            s += args.batch_size


        print('epoch %2d   ' % step, end='')
        train_acc, _ = evaluate(train_entity_pairs, train_labels)
        valid_acc, _ = evaluate(valid_entity_pairs, valid_labels)
        test_acc, test_scores = evaluate(test_entity_pairs, test_labels)

        current_res = 'acc: %.4f' % test_acc
        print("Current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print('train acc: %.4f   valid acc: %.4f   test acc: %.4f' % (train_acc, valid_acc, test_acc))
        mrr, mr, hit1, hit3, hit10 = calculate_ranking_metrics(test_triplets, test_scores, true_relations)
        current_res += '   mrr: %.4f   mr: %.4f   h1: %.4f   h3: %.4f   h10: %.4f' % (mrr, mr, hit1, hit3, hit10)
        print('           mrr: %.4f   mr: %.4f   h1: %.4f   h3: %.4f   h10: %.4f' % (mrr, mr, hit1, hit3, hit10))
        print()

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            final_res = current_res

    print('final results\n%s' % final_res)

def get_feed_dict(entity_pairs, train_edges, labels, start, end):

    feed_dict = {}
    if args.use_context:
        feed_dict["entity_pairs"] = entity_pairs[start:end]
        if train_edges is not None:
            feed_dict["train_edges"] = train_edges[start:end]
        else:

            feed_dict["train_edges"] = torch.LongTensor(np.array([-1] * (end - start), np.int32)).cuda() if args.cuda \
                else torch.LongTensor(np.array([-1] * (end - start), np.int32))
    feed_dict["labels"] = labels[start:end]

    return feed_dict

def evaluate(entity_pairs, labels):

    acc_list = []
    scores_list = []

    s = 0
    while s + args.batch_size <= len(labels):
        acc, scores = model.test_step(model, get_feed_dict(
            entity_pairs, None, labels, s, s + args.batch_size))
        acc_list.append(acc.float().mean().item())
        scores_list.extend(scores.cpu().tolist())
        s += args.batch_size
    return float(np.mean(acc_list)), np.array(scores_list)


def calculate_ranking_metrics(triplets, scores, true_relations):
    for i in range(scores.shape[0]):
        head, tail, relation = triplets[i]
        for j in true_relations.get((head, tail), set()) - {relation}:
            if j < scores.shape[1]:
                scores[i, j] -= 1.0

    sorted_indices = np.argsort(-scores, axis=1)
    relations_list = [trip[2] for trip in triplets]
    relations = np.array(relations_list)

    if relations.shape[0] != sorted_indices.shape[0]:
        relations = relations[:sorted_indices.shape[0]]

    sorted_indices -= np.expand_dims(relations, 1)
    zero_coordinates = np.argwhere(sorted_indices == 0)

    if zero_coordinates.shape[0] != sorted_indices.shape[0]:
        rankings = np.full(scores.shape[0], scores.shape[1])
        found_indices = zero_coordinates[:, 0]
        found_ranks = zero_coordinates[:, 1] + 1
        rankings[found_indices] = found_ranks
    else:
        rankings = zero_coordinates[:, 1] + 1

    mrr = float(np.mean(1.0 / rankings))
    mr = float(np.mean(rankings))
    hit1 = float(np.mean(rankings <= 1))
    hit3 = float(np.mean(rankings <= 3))
    hit10 = float(np.mean(rankings <= 10))

    return mrr, mr, hit1, hit3, hit10
