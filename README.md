# HiProTransfer: Hierarchical Prototype-Guided Anomaly Detection for Distributed Microservice Systems in Production

[English](README.md) | [中文](README_zh.md)

HiProTransfer is a hierarchical prototype-guided anomaly detection method for
distributed microservice systems in production. This repository provides the
implementation, configuration, and anonymized dataset required to run the
training and detection pipeline.

The core model is **MATA**, the **Mask-aware Adversarial Temporal Autoencoder**.
MATA combines a mask-aware temporal encoder, dual reconstruction decoders,
service-level prototypes, and machine-level adaptation.

## Directory Layout

```text
HiProTransfer/
  configs/default.yaml
  data/
  scripts/run_pipeline.sh
  hiprotransfer/
    config.py
    data.py
    hierarchy.py
    evaluation.py
    mata.py
    train_datacenter.py
    train_service.py
    detect_machine.py
```

## Environment

Python 3.7 is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

The dependency versions follow the original experiment environment.

## Data Format

Place datasets under `data/<dataset_name>/`:

```text
data/<dataset_name>/
  offline_data.npy
  online_data.npy
  labels.npy
  machine_list.json
  metric_mask.npy
```

`offline_data.npy` and `online_data.npy` should have shape
`[machines, time, metrics]`. `labels.npy` should have shape `[machines, time]`.
`metric_mask.npy` is a required dataset file and has shape `[machines, metrics]`.

Machine names in `machine_list.json` should follow:

```text
datacenters_x-server_y-machine_z
```

This repository includes `data/data1`, a confidentiality-compliant,
topology-preserving 50-machine subset pruned from the full private dataset.
See `data/README.md` for its release scope, preprocessing details, file schemas,
and topology-field definitions.

## Configuration

Edit `configs/default.yaml` to change dataset paths, model size, training
epochs, output path, and device. The default seed is `2026`.

## Run

Build hierarchy:

```bash
python -m hiprotransfer.hierarchy \
  --machine-list data/data1/machine_list.json \
  --offline-data data/data1/offline_data.npy \
  --metric-mask data/data1/metric_mask.npy \
  --output data/data1/hierarchy.json
```

Train datacenter models:

```bash
python -m hiprotransfer.train_datacenter --config configs/default.yaml
```

Train service models and prototypes:

```bash
python -m hiprotransfer.train_service --config configs/default.yaml
```

Run machine detection:

```bash
python -m hiprotransfer.detect_machine --config configs/default.yaml
```

Or run all steps:

```bash
sh scripts/run_pipeline.sh configs/default.yaml
```

Detection uses configurable sigma thresholding, with the default set to 3-sigma:

```text
threshold = mean(reference_scores) + sigma * std(reference_scores)
```

Results are written under:

```text
outputs/HiProTransfer/<dataset_name>/
```
