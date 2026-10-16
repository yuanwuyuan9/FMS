# Entity Prediction

Run the following commands from the repository root (`FMS/`), unless a directory change is shown.

## Installation

```bash
cd /path/to/FMS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES=0
```

## Transductive Setting

Run a dataset:

```bash
python Entity_Prediction/Transductive/src/run.py --dataset umls
python Entity_Prediction/Transductive/src/run.py --dataset FB15k-237
python Entity_Prediction/Transductive/src/run.py --dataset nell
```

Run all three datasets in sequence:

```bash
python Entity_Prediction/Transductive/src/run.py
```

Save a run to a specified directory:

```bash
python Entity_Prediction/Transductive/src/run.py --dataset umls \
  --output_dir Entity_Prediction/Transductive/results/umls_run1
```

Evaluate the saved checkpoint:

```bash
python Entity_Prediction/Transductive/src/evaluate.py \
  --checkpoint Entity_Prediction/Transductive/results/umls_run1/umls.pt \
  --results_json Entity_Prediction/Transductive/results/umls_run1/evaluation.json
```

## Inductive Setting

Run the preset experiments:

```bash
cd Entity_Prediction/Inductive/src
python run.py
```

Run a single dataset on GPU from this directory:

```bash
python main.py --dataset fb237_v1 --cuda
python main.py --dataset WN18RR_v1 --cuda
python main.py --dataset nell_v1 --cuda
```
