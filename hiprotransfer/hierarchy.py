import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_machine_name(machine_name):
    """Extract identifiers from datacenters_x-server_y-machine_z."""
    match = re.fullmatch(
        r"(datacenters_[1-9][0-9]*)-(server_[1-9][0-9]*)-"
        r"(machine_[1-9][0-9]*)",
        machine_name,
    )
    if match is None:
        raise ValueError(
            "machine_name must follow "
            "datacenters_x-server_y-machine_z format"
        )
    return match.group(1), match.group(2)


def _masked_distance(mean_a, mean_b, mask_a=None, mask_b=None):
    if mask_a is None or mask_b is None:
        return float(np.linalg.norm(mean_a - mean_b))
    common = mask_a.astype(bool) & mask_b.astype(bool)
    if common.sum() == 0:
        return 0.0
    return float(np.linalg.norm(mean_a[common] - mean_b[common]) / common.sum())


def select_center_machine(service_data, service_machine_indices=None, metric_mask=None):
    """Select the center machine by average pairwise mean-series distance."""
    service_data = np.asarray(service_data, dtype=np.float32)
    n_machines = service_data.shape[0]
    if n_machines == 0:
        raise ValueError("service_data must contain at least one machine")
    if n_machines == 1:
        return 0
    if service_machine_indices is None:
        service_machine_indices = list(range(n_machines))

    machine_means = service_data.mean(axis=1)
    distances = np.zeros((n_machines, n_machines), dtype=np.float32)
    for i in range(n_machines):
        for j in range(i + 1, n_machines):
            mask_i = metric_mask[service_machine_indices[i]] if metric_mask is not None else None
            mask_j = metric_mask[service_machine_indices[j]] if metric_mask is not None else None
            dist = _masked_distance(machine_means[i], machine_means[j], mask_i, mask_j)
            distances[i, j] = dist
            distances[j, i] = dist
    return int(np.argmin(distances.sum(axis=1) / float(n_machines - 1)))


def build_hierarchy(machine_list, offline_data=None, metric_mask=None):
    """Build datacenter -> service -> machine hierarchy."""
    grouped = defaultdict(lambda: defaultdict(list))
    for machine_idx, machine_name in enumerate(machine_list):
        datacenter_id, service_id = parse_machine_name(machine_name)
        grouped[datacenter_id][service_id].append({"index": machine_idx, "name": machine_name})

    hierarchy = {"datacenters": {}}

    for datacenter_id, services in grouped.items():
        datacenter = {"services": {}, "machine_count": 0}
        for service_id, machines in services.items():
            machine_indices = [item["index"] for item in machines]
            machine_names = [item["name"] for item in machines]
            if offline_data is not None and len(machine_indices) > 1:
                center_rel = select_center_machine(
                    np.asarray(offline_data)[machine_indices],
                    service_machine_indices=machine_indices,
                    metric_mask=metric_mask,
                )
                center_machine = machine_indices[center_rel]
            else:
                center_machine = machine_indices[0]
            datacenter["services"][service_id] = {
                "machines": machine_indices,
                "machine_names": machine_names,
                "center_machine": int(center_machine),
                "machine_count": len(machine_indices),
            }
            datacenter["machine_count"] += len(machine_indices)
        hierarchy["datacenters"][datacenter_id] = datacenter

    return hierarchy


def main():
    parser = argparse.ArgumentParser(description="Build HiProTransfer hierarchy.")
    parser.add_argument("--machine-list", required=True)
    parser.add_argument("--offline-data", default=None)
    parser.add_argument("--metric-mask", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.machine_list, "r") as f:
        machine_list = json.load(f)
    offline_data = np.load(args.offline_data) if args.offline_data and Path(args.offline_data).exists() else None
    metric_mask = np.load(args.metric_mask)
    hierarchy = build_hierarchy(machine_list, offline_data=offline_data, metric_mask=metric_mask)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(hierarchy, f, indent=2)
    print("Hierarchy saved to {}".format(output_path))


if __name__ == "__main__":
    main()
