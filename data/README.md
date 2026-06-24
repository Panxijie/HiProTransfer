# Data Directory

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

## Public release scope

Due to corporate confidentiality, production-security, and data-governance
restrictions, the complete production dataset cannot be released. This repository
therefore publishes only `data1`, an anonymized, topology-preserving subset pruned
from the full dataset.

The released subset contains 50 machines from 3 datacenters and 6 services. It is
intended to illustrate the datacenter -> service -> machine hierarchy and support
functional reproduction of the HiProTransfer pipeline.

## Included files

```text
data/data1/
  offline_data.npy
  online_data.npy
  labels.npy
  machine_list.json
  metric_mask.npy
  hierarchy.json
```

Topology distribution:

| Datacenter | Service | Machines |
|:---:|:---:|:---:|
| datacenters_1 | server_1 | 12 |
| datacenters_1 | server_2 | 6 |
| datacenters_2 | server_3 | 12 |
| datacenters_2 | server_4 | 8 |
| datacenters_3 | server_2 | 10 |
| datacenters_3 | server_5 | 2 |

## Preprocessing and file definitions

The released arrays contain **preprocessed data**, not raw production metrics.
Offline and online values have already been transformed by the original
per-machine preprocessing procedure using offline-data MinMax scaling.

NumPy arrays require a regular rectangular shape, while different machines may
have different sets of available metrics. To form tensors with a shared
19-dimensional metric axis, unavailable metrics are filled with `0`.
`metric_mask.npy` explicitly records whether each metric is present, allowing the
pipeline to distinguish a missing metric represented by a placeholder zero from a
valid observed metric whose real value is zero. Machine, datacenter, service, and
metric identifiers are anonymized.

### Index alignment

The first dimension of every array uses the same public machine index:

```text
machine_list[i]
offline_data[i, :, :]
online_data[i, :, :]
labels[i, :]
metric_mask[i, :]
```

All machine indices stored in `hierarchy.json` refer to this same 0-based index.
The public indices are newly assigned for this subset and are not the original
indices from the private full dataset.

### `offline_data.npy`

- Shape: `[50, 20160, 19]`
- Type: `float32`
- Axes: `[machine, time point, metric]`
- Coverage: 20,160 one-minute points per machine, corresponding to 14 complete
  days.
- Values: preprocessed per machine using statistics fitted on that machine's
  offline sequence. Valid non-constant metrics are scaled to `[-1, 1]`.
- Purpose: hierarchy construction and datacenter/service/machine training.

### `online_data.npy`

- Shape: `[50, 10080, 19]`
- Type: `float32`
- Axes: `[machine, time point, metric]`
- Coverage: 10,080 one-minute points per machine, corresponding to 7 complete
  days.
- Values: already transformed using the preprocessing parameters fitted from the
  corresponding offline sequence. Values outside the offline range may extend
  beyond `[-1, 1]`; the original preprocessing pipeline clips the released online
  values to the closed interval `[-3, 3]`.
- Purpose: anomaly detection and evaluation.

### `labels.npy`

- Shape: `[50, 10080]`
- Type: `int32`
- Axes: `[machine, online time point]`
- Semantics: `0` means normal and `1` means anomalous.
- Alignment: `labels[i, t]` is the label for `online_data[i, t, :]`.
- The offline data has no anomaly-label file because it is used as the
  unsupervised training/reference period.

### `machine_list.json`

An ordered JSON array containing 50 anonymized machine names. Position `i` is the
public machine index used by every other file.

Machine names follow:

```text
datacenters_<datacenter>-server_<service>-machine_<machine>
```

For example, `datacenters_1-server_1-machine_1` belongs to datacenter
`datacenters_1` and service `server_1`. Datacenter IDs are continuous from 1 to 3,
service IDs are continuous from 1 to 5, and machine IDs are continuous from 1 to
50. These identifiers are anonymous and do not reveal production names.

The `machine_<n>` suffix is a 1-based public label. Array access remains 0-based,
so `machine_1` is stored at array index `0`, `machine_2` at index `1`, and so on.

### `metric_mask.npy`

- Shape: `[50, 19]`
- Type: `uint8`
- Semantics: `metric_mask[i, j] == 1` means metric `j` exists for machine `i`;
  `0` means that metric is unavailable and its data column is zero-filled.
- Usage: training and scoring must exclude unavailable dimensions instead of
  treating their zero-filled values as observations. When the mask is `1`, a data
  value of `0` is a valid preprocessed observation rather than a missing value.
- Metric positions are anonymous indices `0` through `18` and align with the last
  dimension of both data arrays.

### `hierarchy.json`

This file encodes the three-level topology:

```text
datacenter -> service -> machine
```

Top-level structure:

```json
{
  "datacenters": {
    "datacenters_1": {
      "services": {
        "server_1": {
          "machines": [0, 1],
          "machine_names": [
            "datacenters_1-server_1-machine_1",
            "datacenters_1-server_1-machine_2"
          ],
          "center_machine": 1,
          "machine_count": 2
        }
      },
      "machine_count": 2
    }
  }
}
```

Field meanings:

- `datacenters`: mapping from anonymized datacenter ID to its services.
- `services`: mapping from anonymized service ID to machines in that service.
- `machines`: public 0-based machine indices. These directly index all `.npy`
  arrays and `machine_list.json`.
- `machine_names`: names corresponding position-by-position to `machines`.
- The numeric suffix in a machine name is 1-based, while values in `machines`
  and `center_machine` are 0-based array indices.
- `center_machine`: public index of the service's representative machine. For
  every candidate machine, the code calculates its average distance to all other
  machines in the same service, using only metrics available on both machines.
  The machine with the smallest average distance is selected because it is the
  center machine of that service. It is not a service-relative list position and
  is not selected by anomaly-detection performance.
- Service-level `machine_count`: number of machines in that service.
- Datacenter-level `machine_count`: sum of machines across its services.

The lists are mutually consistent:

```text
machine_names[k] == machine_list[machines[k]]
```

## Loading example

```python
import json
import numpy as np

root = "data/data1"

offline = np.load(f"{root}/offline_data.npy")
online = np.load(f"{root}/online_data.npy")
labels = np.load(f"{root}/labels.npy")
metric_mask = np.load(f"{root}/metric_mask.npy")

with open(f"{root}/machine_list.json", "r") as f:
    machine_list = json.load(f)
with open(f"{root}/hierarchy.json", "r") as f:
    hierarchy = json.load(f)

machine_idx = 0
print(machine_list[machine_idx])
print(offline[machine_idx].shape)     # (20160, 19)
print(online[machine_idx].shape)      # (10080, 19)
print(labels[machine_idx].shape)      # (10080,)
print(metric_mask[machine_idx])       # 19-dimensional availability mask
```

## Execution

Run from the repository root:

```bash
sh scripts/run_pipeline.sh configs/default.yaml data/data1
```
