import argparse
from pathlib import Path

import yaml


class ConfigNode(dict):
    """Dictionary with attribute access for YAML configuration."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def _to_node(value):
    if isinstance(value, dict):
        return ConfigNode({key: _to_node(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_node(item) for item in value]
    return value


def load_config(config_path):
    config_path = Path(config_path).resolve()
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    cfg = _to_node(raw)
    cfg.config_path = config_path
    cfg.root_dir = config_path.parents[1]
    return cfg


def resolve_path(cfg, path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return cfg.root_dir / path


def dataset_dir(cfg):
    return resolve_path(cfg, cfg.dataset.data_dir)


def output_dir(cfg):
    return resolve_path(cfg, cfg.output.output_dir) / "HiProTransfer" / cfg.dataset.name


def common_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    return parser
