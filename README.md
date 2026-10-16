# Flow-Modulated Scoring for Semantic-Aware Knowledge Graph Completion

PyTorch implementation of **Flow-Modulated Scoring for Semantic-Aware Knowledge Graph Completion**.

**[arXiv Paper](https://arxiv.org/abs/2506.23137)**

## Overview

Flow-Modulated Scoring (FMS) combines semantic-aware relational context with conditional flow modeling for knowledge graph completion. FMS learns continuous transformations of contextual representations and uses the resulting features to score candidate entities or relations.

Training and evaluation workflows for entity prediction and temporal entity prediction follow [RED-GNN](https://github.com/LARS-research/RED-GNN). Relation prediction follows the training and evaluation workflow of [PathCon](https://github.com/hwwang55/PathCon).

FMS is applied to the three tasks as follows:

- **Entity Prediction:** FMS encodes the head entity's relational context and uses flow conditioned on the query relation to construct latent features for scoring all candidate entities.
- **Temporal Entity Prediction:** FMS encodes time-aware relational context from past events and uses query-conditioned flow to construct latent features for scoring future entity candidates.
- **Relation Prediction:** FMS encodes the relational contexts of both entities and learns a flow between their contextual representations to modulate relation scores.

Run the following commands from the repository root (`FMS/`).

## Installation

```bash
cd /path/to/FMS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES=0
```

## Running Experiments

### Entity Prediction

```bash
python Entity_Prediction/Transductive/src/run.py --dataset umls
```

See [Entity Prediction](Entity_Prediction/README.md) for all running commands.

### Temporal Entity Prediction

```bash
python Entity_Prediction_TKG/src/run.py --dataset ICEWS14
```

See [Temporal Entity Prediction](Entity_Prediction_TKG/README.md) for all running commands.

### Relation Prediction

```bash
python Relation_Prediction/Transductive/src/run.py
```

See [Relation Prediction](Relation_Prediction/README.md) for transductive and inductive running commands.

## License

[MIT License](LICENSE).
