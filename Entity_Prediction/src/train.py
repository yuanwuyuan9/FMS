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

    train_labels = torch.LongTensor(np.array([triplet[1] for triplet in train_triplets], np.int32))
    valid_labels = torch.LongTensor(np.array([triplet[1] for triplet in valid_triplets], np.int32))
    test_labels = torch.LongTensor(np.array([triplet[1] for triplet in test_triplets], np.int32))

    train_relations = torch.LongTensor(np.array([triplet[2] for triplet in train_triplets], np.int32))
    valid_relations = torch.LongTensor(np.array([triplet[2] for triplet in valid_triplets], np.int32))
    test_relations = torch.LongTensor(np.array([triplet[2] for triplet in test_triplets], np.int32))

    model = FMS(args, n_relations, neighbor_params, n_entity)
    
    model = model.cuda()
    train_labels = train_labels.cuda()
    valid_labels = valid_labels.cuda()
    test_labels = test_labels.cuda()
    if args.use_context:
        train_edges = train_edges.cuda()
        train_entity_pairs = train_entity_pairs.cuda()
        valid_entity_pairs = valid_entity_pairs.cuda()
        test_entity_pairs = test_entity_pairs.cuda()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.l2,
    )

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}, Requires Grad: {param.requires_grad}")
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    

    true_relations = defaultdict(set)
    for head, tail, relation in train_triplets + valid_triplets + test_triplets:
        true_relations[(head, tail)].add(relation)

    true_ent = defaultdict(set)
    for head, tail, relation in train_triplets + valid_triplets + test_triplets:
        true_ent[(head, relation)].add(tail)

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
                train_entity_pairs, train_edges, train_labels, train_relations, s, s + args.batch_size))
            s += args.batch_size


        print('epoch %2d   ' % step, end='')

        model.eval()
        with torch.no_grad():
            # Use the new unified evaluation function
            train_acc, _ = evaluate_and_rank(
                train_entity_pairs, train_labels, train_relations,
                train_triplets, true_ent, is_test_set=False
            )
            valid_acc, _ = evaluate_and_rank(
                valid_entity_pairs, valid_labels, valid_relations,
                valid_triplets, true_ent, is_test_set=False
            )
            test_acc, ranking_metrics = evaluate_and_rank(
                test_entity_pairs, test_labels, test_relations,
                test_triplets, true_ent, is_test_set=True
            )

        # Unpack the ranking metrics
        mrr, mr, hit1, hit3, hit10 = ranking_metrics

        # Update printing and logging
        current_res = 'acc: %.4f' % test_acc
        print("Current time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print('train acc: %.4f   valid acc: %.4f   test acc: %.4f' % (train_acc, valid_acc, test_acc))
        
        current_res += '   mrr: %.4f   mr: %.4f   h1: %.4f   h3: %.4f   h10: %.4f' % (mrr, mr, hit1, hit3, hit10)
        print('           mrr: %.4f   mr: %.4f   h1: %.4f   h3: %.4f   h10: %.4f' % (mrr, mr, hit1, hit3, hit10))
        print()

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            final_res = current_res

    print('final results\n%s' % final_res)



def get_feed_dict(entity_pairs, train_edges, labels, relations, start, end):

    feed_dict = {}
    if args.use_context:
        feed_dict["entity_pairs"] = entity_pairs[start:end]
        if train_edges is not None:
            feed_dict["train_edges"] = train_edges[start:end]
        else:
            feed_dict["train_edges"] = torch.LongTensor(np.array([-1] * (end - start), np.int32)).cuda() if args.cuda \
                else torch.LongTensor(np.array([-1] * (end - start), np.int32))

    feed_dict["relations"] = relations[start:end]
    feed_dict["labels"] = labels[start:end]

    return feed_dict


def evaluate_and_rank(entity_pairs, labels, relations, triplets, true_ent, is_test_set=False):
    """
    Performs evaluation and ranking in batches to avoid memory explosion.
    
    Args:
        entity_pairs: The (head, tail) entity pairs for the model input.
        labels: The true entity labels.
        relations: The relation edge in a triple
        triplets: The original (h, t, r) triplets corresponding to the data.
        true_ent: A dictionary for filtering false negatives during ranking.
        is_test_set: A boolean flag. If True, calculate and return ranking metrics.

    Returns:
        A tuple containing:
        - final_accuracy (float)
        - If is_test_set is True, also returns (mrr, mr, hit1, hit3, hit10)
        - Otherwise, returns None for the ranking metrics part.
    """
    total_correct = 0
    total_samples = 0
    all_rankings = []
    
    s = 0
    while s + args.batch_size <= len(labels):
        end = s + args.batch_size
        batch = get_feed_dict(entity_pairs, None, labels, relations, s, end)
        correct_tensor, scores_tensor = model.test_step(model, batch)

        total_correct += correct_tensor.sum().item()
        total_samples += len(correct_tensor)
        if is_test_set:
            batch_triplets = triplets[s:end]

            for i in range(scores_tensor.shape[0]):
                head, _ , relation = batch_triplets[i]
                true_tail = batch_triplets[i][1]
                false_negatives = torch.LongTensor(list(true_ent.get((head, relation), set()) - {true_tail}))
                if len(false_negatives) > 0:
                    scores_tensor[i, false_negatives.to(scores_tensor.device)] = -1e9

            true_tails = torch.LongTensor([triplet[1] for triplet in batch_triplets]).to(scores_tensor.device)
            true_tail_scores = scores_tensor.gather(1, true_tails.unsqueeze(1))
            ranks_tensor = (scores_tensor > true_tail_scores).sum(dim=1) + 1
            all_rankings.append(ranks_tensor.cpu())

        s += args.batch_size
    final_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    if not is_test_set:
        return final_accuracy, None
    if not all_rankings:
        return final_accuracy, (0.0, 0.0, 0.0, 0.0, 0.0)

    rankings = torch.cat(all_rankings, dim=0).numpy().astype(float)
    
    mrr = float(np.mean(1.0 / rankings))
    mr = float(np.mean(rankings))
    hit1 = float(np.mean(rankings <= 1))
    hit3 = float(np.mean(rankings <= 3))
    hit10 = float(np.mean(rankings <= 10))

    return final_accuracy, (mrr, mr, hit1, hit3, hit10)

