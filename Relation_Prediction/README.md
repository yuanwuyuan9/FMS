# Relation Prediction

Run the following commands from the repository root (`FMS/`).

## Installation

```bash
cd /path/to/FMS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES=0
```

## Transductive Setting

```bash
python Relation_Prediction/Transductive/src/run.py
```

## Inductive Setting

```bash
python Relation_Prediction/Inductive/src/run.py
```
