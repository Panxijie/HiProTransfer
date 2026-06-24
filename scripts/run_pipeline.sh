#!/bin/sh
set -eu

CONFIG="${1:-configs/default.yaml}"
DATA_DIR="${2:-data/data1}"

for file in offline_data.npy online_data.npy labels.npy machine_list.json metric_mask.npy; do
  if [ ! -f "${DATA_DIR}/${file}" ]; then
    echo "Missing required dataset file: ${DATA_DIR}/${file}"
    exit 1
  fi
done

python -m hiprotransfer.hierarchy \
  --machine-list "${DATA_DIR}/machine_list.json" \
  --offline-data "${DATA_DIR}/offline_data.npy" \
  --metric-mask "${DATA_DIR}/metric_mask.npy" \
  --output "${DATA_DIR}/hierarchy.json"

python -m hiprotransfer.train_datacenter --config "$CONFIG"
python -m hiprotransfer.train_service --config "$CONFIG"
python -m hiprotransfer.detect_machine --config "$CONFIG"
