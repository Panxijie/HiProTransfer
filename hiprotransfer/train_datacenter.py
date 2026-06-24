import json
from pathlib import Path

import numpy as np

from .config import common_arg_parser, dataset_dir, load_config, output_dir, resolve_path
from .data import preprocess_minmax, set_seed, split_day_segments
from .mata import MATA


def _load_hierarchy(cfg):
    hierarchy_path = resolve_path(cfg, cfg.dataset.hierarchy)
    with open(hierarchy_path, "r") as f:
        return json.load(f)


def _load_metric_mask(cfg):
    path = dataset_dir(cfg) / cfg.dataset.metric_mask
    if not path.exists():
        raise FileNotFoundError("Missing required metric mask: {}".format(path))
    return np.load(path)


def _normalize_machines(raw_data):
    normalized = np.zeros_like(raw_data, dtype=np.float32)
    for idx in range(raw_data.shape[0]):
        normalized[idx], _ = preprocess_minmax(raw_data[idx])
    return normalized


def _sample_datacenter_segments(datacenter_info, offline_data, cfg, metric_mask):
    minutes_per_day = int(cfg.hierarchy.minutes_per_day)
    days_per_machine = int(cfg.hierarchy.days_per_machine)
    budget = int(cfg.hierarchy.datacenter_samples)
    min_per_service = int(cfg.hierarchy.min_samples_per_service)
    cap_per_service = int(cfg.hierarchy.cap_samples_per_service)

    segments = []
    masks = []
    for service_id, service_info in sorted(datacenter_info["services"].items()):
        machines = service_info["machines"]
        count = min(max(min_per_service, len(machines)), cap_per_service)
        candidates = [(machine_idx, day_idx) for machine_idx in machines for day_idx in range(days_per_machine)]
        if not candidates:
            continue
        selected = np.random.choice(len(candidates), size=min(count, len(candidates)), replace=False)
        for candidate_idx in selected:
            machine_idx, day_idx = candidates[candidate_idx]
            start = day_idx * minutes_per_day
            end = start + minutes_per_day
            segments.append(offline_data[machine_idx, start:end, :])
            if metric_mask is not None:
                masks.append(metric_mask[machine_idx].astype(np.float32))

    if len(segments) > budget:
        selected = np.random.choice(len(segments), size=budget, replace=False)
        segments = [segments[idx] for idx in selected]
        masks = [masks[idx] for idx in selected] if masks else []
    return segments, masks if masks else None


def train_datacenter_models(cfg):
    set_seed(int(cfg.training.seed))
    hierarchy = _load_hierarchy(cfg)
    raw_data = np.load(dataset_dir(cfg) / cfg.dataset.offline_data).astype(np.float32)
    offline_data = _normalize_machines(raw_data)
    metric_mask = _load_metric_mask(cfg)
    save_root = output_dir(cfg) / "datacenter_models"
    save_root.mkdir(parents=True, exist_ok=True)

    for datacenter_id, datacenter_info in sorted(hierarchy["datacenters"].items()):
        segments, masks = _sample_datacenter_segments(datacenter_info, offline_data, cfg, metric_mask)
        if not segments:
            print("Skip {}: no training segments".format(datacenter_id))
            continue
        model = MATA(
            feature_dim=int(cfg.model.feature_dim),
            window_size=int(cfg.model.window_size),
            latent_dim=int(cfg.model.latent_dim),
            batch_size=int(cfg.training.batch_size),
            hidden_channels=int(cfg.model.tcn_hidden_channels),
            kernel_size=int(cfg.model.tcn_kernel_size),
            lr=float(cfg.training.lr),
            device=cfg.training.device,
        )
        model.fit(
            segments,
            epochs=int(cfg.training.datacenter_epochs),
            metric_masks=masks,
            progress_desc="datacenter {}".format(datacenter_id),
        )
        model_dir = save_root / datacenter_id
        model.save(model_dir)
        print("Saved datacenter model to {}".format(model_dir))


def main():
    parser = common_arg_parser("Train HiProTransfer datacenter models.")
    args = parser.parse_args()
    train_datacenter_models(load_config(args.config))


if __name__ == "__main__":
    main()
