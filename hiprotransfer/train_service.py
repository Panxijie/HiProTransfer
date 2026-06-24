import json
from pathlib import Path

import numpy as np

from .config import common_arg_parser, dataset_dir, load_config, output_dir, resolve_path
from .data import preprocess_minmax, set_seed, split_day_segments
from .hierarchy import _masked_distance, select_center_machine
from .mata import MATA


def _normalize_machines(raw_data):
    normalized = np.zeros_like(raw_data, dtype=np.float32)
    for idx in range(raw_data.shape[0]):
        normalized[idx], _ = preprocess_minmax(raw_data[idx])
    return normalized


def _load_metric_mask(cfg):
    path = dataset_dir(cfg) / cfg.dataset.metric_mask
    if not path.exists():
        raise FileNotFoundError("Missing required metric mask: {}".format(path))
    return np.load(path)


def _make_model(cfg):
    return MATA(
        feature_dim=int(cfg.model.feature_dim),
        window_size=int(cfg.model.window_size),
        latent_dim=int(cfg.model.latent_dim),
        batch_size=int(cfg.training.batch_size),
        hidden_channels=int(cfg.model.tcn_hidden_channels),
        kernel_size=int(cfg.model.tcn_kernel_size),
        lr=float(cfg.training.lr),
        device=cfg.training.device,
    )


def _build_service_prototype(model, service_data, service_machines, center_relative_idx, metric_mask, cfg):
    center_idx = service_machines[center_relative_idx]
    center_mask = metric_mask[center_idx] if metric_mask is not None else None
    latents = [model.latent(service_data[center_relative_idx], metric_mask=center_mask)]
    weights = [float(cfg.prototype.center_weight)]

    center_mean = service_data[center_relative_idx].mean(axis=0)
    distances = []
    for rel_idx, machine_idx in enumerate(service_machines):
        if rel_idx == center_relative_idx:
            continue
        mask_i = metric_mask[machine_idx] if metric_mask is not None else None
        dist = _masked_distance(service_data[rel_idx].mean(axis=0), center_mean, mask_i, center_mask)
        distances.append((dist, rel_idx, machine_idx))
    distances.sort(key=lambda item: item[0])

    neighbor_count = int(cfg.prototype.neighbors)
    selected = distances[:neighbor_count]
    neighbor_weight = (1.0 - float(cfg.prototype.center_weight)) / max(len(selected), 1)
    for _, rel_idx, machine_idx in selected:
        mask_i = metric_mask[machine_idx] if metric_mask is not None else None
        latents.append(model.latent(service_data[rel_idx], metric_mask=mask_i))
        weights.append(neighbor_weight)

    prototype = np.zeros((int(cfg.model.latent_dim),), dtype=np.float32)
    for latent, weight in zip(latents, weights):
        prototype += latent.mean(axis=0) * weight
    return prototype


def train_service_models(cfg):
    set_seed(int(cfg.training.seed))
    hierarchy_path = resolve_path(cfg, cfg.dataset.hierarchy)
    with open(hierarchy_path, "r") as f:
        hierarchy = json.load(f)
    raw_data = np.load(dataset_dir(cfg) / cfg.dataset.offline_data).astype(np.float32)
    offline_data = _normalize_machines(raw_data)
    metric_mask = _load_metric_mask(cfg)
    model_root = output_dir(cfg)
    datacenter_root = model_root / "datacenter_models"
    service_root = model_root / "service_models"
    service_root.mkdir(parents=True, exist_ok=True)

    for datacenter_id, datacenter_info in sorted(hierarchy["datacenters"].items()):
        datacenter_model_dir = datacenter_root / datacenter_id
        if not datacenter_model_dir.exists():
            print("Skip {}: missing datacenter model".format(datacenter_id))
            continue
        for service_id, service_info in sorted(datacenter_info["services"].items()):
            service_machines = service_info["machines"]
            service_data = offline_data[service_machines]
            center_relative_idx = select_center_machine(
                service_data,
                service_machine_indices=service_machines,
                metric_mask=metric_mask,
            )
            center_idx = service_machines[center_relative_idx]
            center_segments = split_day_segments(
                service_data[center_relative_idx],
                minutes_per_day=int(cfg.hierarchy.minutes_per_day),
                days=int(cfg.hierarchy.days_per_machine),
            )
            center_mask = metric_mask[center_idx] if metric_mask is not None else None
            model = _make_model(cfg)
            model.restore(datacenter_model_dir)
            model.fit(
                center_segments,
                epochs=int(cfg.training.service_epochs),
                metric_masks=[center_mask] * len(center_segments) if center_mask is not None else None,
                progress_desc="service {}/{}".format(datacenter_id, service_id),
            )
            service_dir = service_root / datacenter_id / service_id
            service_dir.mkdir(parents=True, exist_ok=True)
            model.save(service_dir)
            prototype = _build_service_prototype(model, service_data, service_machines, center_relative_idx, metric_mask, cfg)
            np.save(service_dir / "service_prototype.npy", prototype)
            strategy = {}
            for machine_idx in service_machines:
                strategy[str(machine_idx)] = {
                    "strategy": "reuse" if machine_idx == center_idx else "finetune",
                    "epochs": 0 if machine_idx == center_idx else int(cfg.training.machine_epochs),
                }
            with open(service_dir / "transfer_strategy.json", "w") as f:
                json.dump(strategy, f, indent=2)
            print("Saved service model to {}".format(service_dir))


def main():
    parser = common_arg_parser("Train HiProTransfer service models.")
    args = parser.parse_args()
    train_service_models(load_config(args.config))


if __name__ == "__main__":
    main()
