# Temporal Entity Prediction

Run the following commands from the repository root (`FMS/`).

## Installation

```bash
cd /path/to/FMS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES=0
```

## Training

Run a dataset:

```bash
python Entity_Prediction_TKG/src/run.py --dataset ICEWS14
python Entity_Prediction_TKG/src/run.py --dataset ICEWS05-15
python Entity_Prediction_TKG/src/run.py --dataset ICEWS18
python Entity_Prediction_TKG/src/run.py --dataset GDELT
python Entity_Prediction_TKG/src/run.py --dataset WIKI
python Entity_Prediction_TKG/src/run.py --dataset YAGO
```

Run all six datasets in sequence:

```bash
python Entity_Prediction_TKG/src/run.py --dataset ICEWS14 ICEWS05-15 ICEWS18 GDELT WIKI YAGO
```

Save a run to a specified directory:

```bash
python Entity_Prediction_TKG/src/run.py --dataset ICEWS14 \
  --output_dir Entity_Prediction_TKG/results/icews14_run1
```

## Checkpoint Evaluation

```bash
python Entity_Prediction_TKG/src/evaluate.py \
  --checkpoint Entity_Prediction_TKG/results/icews14_run1/ICEWS14.pt
```
