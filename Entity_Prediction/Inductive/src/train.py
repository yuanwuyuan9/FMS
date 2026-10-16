from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F

from data_loader import PREDICT_TAIL, PREDICT_HEAD
from model import FMS


args = None
model = None


def _to_tensor_examples(examples):
    return {
        "known_entities": torch.LongTensor(examples["known_entities"]),
        "target_entities": torch.LongTensor(examples["target_entities"]),
        "relations": torch.LongTensor(examples["relations"]),
        "mask_edges": torch.LongTensor(examples["mask_edges"]),
        "target_sides": examples["target_sides"],
        "source_triplets": examples["source_triplets"],
    }


def _move_if_cuda(tensor):
    return tensor.cuda() if args.cuda else tensor


def _move_examples(examples):
    moved = {}
    for key, value in examples.items():
        if torch.is_tensor(value):
            moved[key] = _move_if_cuda(value)
        else:
            moved[key] = value
    return moved


def train(model_args, data):
    global args, model
    args = model_args

    train_examples = _move_examples(_to_tensor_examples(data["examples"]["train"]))
    base_valid_examples = _move_examples(_to_tensor_examples(data["examples"]["base_valid"]))
    ind_valid_examples = _move_examples(_to_tensor_examples(data["examples"]["ind_valid"]))
    ind_test_examples = _move_examples(_to_tensor_examples(data["examples"]["ind_test"]))
    ind_all_examples = _move_examples(_to_tensor_examples(data["examples"]["ind_all"]))

    model = FMS(
        args,
        data["n_relations"],
        data["graphs"]["base"],
        data["n_entities"]["base"],
    )
    if args.cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.l2,
    )

    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print("start inductive entity prediction training ...")
    print(
        "protocol: base train.txt is support graph and masked train query; "
        "base valid.txt is extra train query; base test.txt is base-valid; "
        "*_ind train.txt is inductive support graph; *_ind valid/test are reported separately."
    )
    print("entity scoring: RED-GNN-style query propagation with graph-level entity context and FMS query context.")
    print(f"selection split: {args.select_split}")

    best_selection = -1.0
    final_res = None

    for epoch in range(args.epoch):
        model.set_context_graph(data["graphs"]["base"], data["n_entities"]["base"])
        train_loss, train_entity_loss, train_cfm_loss = run_train_epoch(
            train_examples,
            data["n_entities"]["base"],
            optimizer,
        )

        base_metrics = evaluate_split(
            base_valid_examples,
            data["graphs"]["base"],
            data["n_entities"]["base"],
            data["true_entities"]["base"],
        )
        ind_valid_metrics = evaluate_split(
            ind_valid_examples,
            data["graphs"]["inductive"],
            data["n_entities"]["inductive"],
            data["true_entities"]["inductive"],
        )
        ind_test_metrics = evaluate_split(
            ind_test_examples,
            data["graphs"]["inductive"],
            data["n_entities"]["inductive"],
            data["true_entities"]["inductive"],
        )
        ind_all_metrics = evaluate_split(
            ind_all_examples,
            data["graphs"]["inductive"],
            data["n_entities"]["inductive"],
            data["true_entities"]["inductive"],
        )

        print("epoch %2d   Current time: %s" % (epoch, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        print("loss: %.4f   entity_loss: %.4f   cfm_loss: %.4f" % (train_loss, train_entity_loss, train_cfm_loss))
        print_metrics("base-valid", base_metrics)
        print_metrics("ind-valid ", ind_valid_metrics)
        print_metrics("ind-test  ", ind_test_metrics)
        print_metrics("ind-all   ", ind_all_metrics)
        print()

        selection_metric = base_metrics["both"][0] if args.select_split == "base-valid" else ind_valid_metrics["both"][0]
        if selection_metric > best_selection:
            best_selection = selection_metric
            final_res = format_metrics(ind_test_metrics)

    print("final inductive test results selected by %s\n%s" % (args.select_split, final_res))


def run_train_epoch(examples, n_entity, optimizer):
    index = torch.randperm(len(examples["target_entities"]), device=examples["target_entities"].device)
    shuffled = {
        key: value[index] if torch.is_tensor(value) else value
        for key, value in examples.items()
        if key not in {"target_sides", "source_triplets"}
    }

    total_loss = 0.0
    total_entity_loss = 0.0
    total_cfm_loss = 0.0
    total_batches = 0
    start = 0
    while start < len(shuffled["target_entities"]):
        end = min(start + args.batch_size, len(shuffled["target_entities"]))
        model.train()
        optimizer.zero_grad()
        scores, cfm_loss = score_query_tensor(shuffled, start, end, n_entity, require_grad=True)
        targets = shuffled["target_entities"][start:end].to(scores.device)
        loss_entity = F.cross_entropy(scores, targets)
        loss = loss_entity + args.lamda * cfm_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_entity_loss += loss_entity.item()
        total_cfm_loss += cfm_loss.item()
        total_batches += 1
        start = end

    denominator = max(total_batches, 1)
    return total_loss / denominator, total_entity_loss / denominator, total_cfm_loss / denominator


def evaluate_split(examples, context_graph, n_entity, true_entities):
    model.set_context_graph(context_graph, n_entity)
    rankings = []
    tail_rankings = []
    head_rankings = []

    model.eval()
    with torch.no_grad():
        total = len(examples["target_entities"])
        start = 0
        while start < total:
            end = min(start + args.eval_query_batch, total)
            scores_batch, _ = score_query_tensor(examples, start, end, n_entity, require_grad=False)
            scores_batch = scores_batch.detach().cpu().numpy()

            for local_idx, example_idx in enumerate(range(start, end)):
                known = int(examples["known_entities"][example_idx].item())
                target = int(examples["target_entities"][example_idx].item())
                relation = int(examples["relations"][example_idx].item())
                target_side = examples["target_sides"][example_idx]

                scores = scores_batch[local_idx]
                filter_entities = true_entities.get((known, relation), set()) - {target}
                if filter_entities:
                    valid_filter = [entity for entity in filter_entities if entity < n_entity]
                    if valid_filter:
                        scores[valid_filter] = -np.inf

                rank = int((scores > scores[target]).sum() + 1)
                rankings.append(rank)
                if target_side == PREDICT_TAIL:
                    tail_rankings.append(rank)
                elif target_side == PREDICT_HEAD:
                    head_rankings.append(rank)
            start = end

    return {
        "both": calculate_metrics(rankings),
        "tail": calculate_metrics(tail_rankings),
        "head": calculate_metrics(head_rankings),
    }


def score_query_tensor(examples, query_start, query_end, n_entity, require_grad):
    device = examples["target_entities"].device

    known = examples["known_entities"][query_start:query_end].to(device)
    relations_query = examples["relations"][query_start:query_end].to(device)
    mask_edges_query = examples["mask_edges"][query_start:query_end].to(device)

    batch = {
        "known_entities": known,
        "relations": relations_query,
        "train_edges": mask_edges_query,
    }
    if "target_entities" in examples:
        batch["target_entities"] = examples["target_entities"][query_start:query_end].to(device)
    original_batch_size = model.batch_size
    model.batch_size = query_end - query_start
    try:
        model(batch)
        scores = model.scores
        cfm_loss = model.loss_cfm
    finally:
        model.batch_size = original_batch_size

    if scores.shape[1] != n_entity:
        raise ValueError(f"Model returned {scores.shape[1]} entity scores, expected {n_entity}.")
    if not require_grad:
        scores = scores.detach()
        cfm_loss = cfm_loss.detach()
    return scores, cfm_loss


def calculate_metrics(rankings):
    if not rankings:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(rankings, dtype=float)
    return (
        float(np.mean(1.0 / arr)),
        float(np.mean(arr)),
        float(np.mean(arr <= 1)),
        float(np.mean(arr <= 3)),
        float(np.mean(arr <= 10)),
    )


def format_metrics(metrics):
    mrr, mr, h1, h3, h10 = metrics["both"]
    tail_mrr, tail_mr, tail_h1, tail_h3, tail_h10 = metrics["tail"]
    head_mrr, head_mr, head_h1, head_h3, head_h10 = metrics["head"]
    return (
        "both mrr: %.4f   mr: %.4f   h1: %.4f   h3: %.4f   h10: %.4f\n"
        "tail mrr: %.4f   mr: %.4f   h1: %.4f   h3: %.4f   h10: %.4f\n"
        "head mrr: %.4f   mr: %.4f   h1: %.4f   h3: %.4f   h10: %.4f"
        % (mrr, mr, h1, h3, h10, tail_mrr, tail_mr, tail_h1, tail_h3, tail_h10, head_mrr, head_mr, head_h1, head_h3, head_h10)
    )


def print_metrics(name, metrics):
    mrr, mr, h1, h3, h10 = metrics["both"]
    tail_mrr, _, _, _, _ = metrics["tail"]
    head_mrr, _, _, _, _ = metrics["head"]
    print(
        "%s both mrr: %.4f   mr: %.4f   h1: %.4f   h3: %.4f   h10: %.4f   tail_mrr: %.4f   head_mrr: %.4f"
        % (name, mrr, mr, h1, h3, h10, tail_mrr, head_mrr)
    )
