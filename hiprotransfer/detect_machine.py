import csv
import json

import numpy as np

from .config import common_arg_parser, dataset_dir, load_config, output_dir, resolve_path
from .data import preprocess_minmax, set_seed
from .evaluation import point_adjust, precision_recall_f1, predict_by_three_sigma
from .mata import MATA


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


def _normalize_pair(offline, online):
    train_scaled, test_scaled = preprocess_minmax(offline, online)
    return train_scaled, test_scaled


def _find_service(hierarchy, machine_idx):
    for datacenter_id, datacenter_info in hierarchy["datacenters"].items():
        for service_id, service_info in datacenter_info["services"].items():
            if machine_idx in service_info["machines"]:
                return datacenter_id, service_id, service_info
    raise KeyError("machine index {} not found in hierarchy".format(machine_idx))


def detect_machines(cfg):
    set_seed(int(cfg.training.seed))
    with open(resolve_path(cfg, cfg.dataset.hierarchy), "r") as f:
        hierarchy = json.load(f)
    raw_offline = np.load(dataset_dir(cfg) / cfg.dataset.offline_data).astype(np.float32)
    raw_online = np.load(dataset_dir(cfg) / cfg.dataset.online_data).astype(np.float32)
    labels = np.load(dataset_dir(cfg) / cfg.dataset.labels).astype(np.int32)
    metric_mask = _load_metric_mask(cfg)
    model_root = output_dir(cfg)
    service_root = model_root / "service_models"
    result_path = model_root / "machine_detection_results.csv"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for machine_idx in range(raw_online.shape[0]):
        datacenter_id, service_id, _ = _find_service(hierarchy, machine_idx)
        service_dir = service_root / datacenter_id / service_id
        model = _make_model(cfg)
        model.restore(service_dir)
        train_values, test_values = _normalize_pair(raw_offline[machine_idx], raw_online[machine_idx])
        machine_mask = metric_mask[machine_idx] if metric_mask is not None else None

        strategy_path = service_dir / "transfer_strategy.json"
        if strategy_path.exists():
            with open(strategy_path, "r") as f:
                strategy = json.load(f).get(str(machine_idx), {})
            epochs = int(strategy.get("epochs", int(cfg.training.machine_epochs)))
        else:
            epochs = int(cfg.training.machine_epochs)
        if epochs > 0:
            prototype_path = service_dir / "service_prototype.npy"
            prototype = np.load(prototype_path) if prototype_path.exists() else None
            model.fit(
                train_values,
                epochs=epochs,
                metric_masks=machine_mask,
                prototype=prototype,
                proto_reg_strength=float(cfg.training.prototype_regularization),
                progress_desc="machine {}".format(machine_idx),
            )

        reference_scores = model.score(train_values, metric_mask=machine_mask)
        test_scores = model.score(test_values, metric_mask=machine_mask)
        predictions, threshold = predict_by_three_sigma(
            test_scores,
            reference_scores,
            sigma=float(cfg.detection.sigma),
        )
        label = labels[machine_idx][-len(predictions):]
        adjusted = point_adjust(label, predictions) if bool(cfg.detection.point_adjust) else predictions
        metrics = precision_recall_f1(label, adjusted)
        rows.append(
            {
                "machine_idx": machine_idx,
                "datacenter_id": datacenter_id,
                "service_id": service_id,
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
            }
        )
        print("machine {} f1={:.4f}".format(machine_idx, metrics["f1"]))

    with open(result_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["machine_idx"])
        writer.writeheader()
        writer.writerows(rows)
    print("Saved detection results to {}".format(result_path))


def main():
    parser = common_arg_parser("Run HiProTransfer machine detection.")
    args = parser.parse_args()
    detect_machines(load_config(args.config))


if __name__ == "__main__":
    main()
