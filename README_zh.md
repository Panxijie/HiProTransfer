# HiProTransfer: Hierarchical Prototype-Guided Anomaly Detection for Distributed Microservice Systems in Production

HiProTransfer 是一种面向生产环境分布式微服务系统的层次化原型引导异常检测方法。本仓库提供运行训练与检测流水线所需的实现代码、配置和匿名化数据集。

核心模型是 **MATA**，即 **Mask-aware Adversarial Temporal Autoencoder**。MATA 结合了 mask-aware 时序编码器、双重重构解码器、服务级原型和机器级自适应。

## 目录结构

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

## 环境

推荐使用 Python 3.7。安装依赖：

```bash
pip install -r requirements.txt
```

依赖版本与原始实验环境保持一致。

## 数据格式

将数据集放在 `data/<dataset_name>/` 下：

```text
data/<dataset_name>/
  offline_data.npy
  online_data.npy
  labels.npy
  machine_list.json
  metric_mask.npy
```

`offline_data.npy` 和 `online_data.npy` 的形状应为 `[machines, time, metrics]`。`labels.npy` 的形状应为 `[machines, time]`。`metric_mask.npy` 是必需的数据文件，形状为 `[machines, metrics]`。

`machine_list.json` 中的机器名应遵循以下格式：

```text
datacenters_x-server_y-machine_z
```

本仓库已包含 `data/data1`：受保密约束、从全量私有数据中剪枝得到且保留拓扑关系的 50 台机器开源子集。其开源范围、预处理方式、文件结构和拓扑字段说明详见 `data/README_zh.md`。

## 配置

修改 `configs/default.yaml` 可调整数据集路径、模型规模、训练轮数、输出路径和设备。默认随机种子为 `2026`。

## 运行

构建层次结构：

```bash
python -m hiprotransfer.hierarchy \
  --machine-list data/data1/machine_list.json \
  --offline-data data/data1/offline_data.npy \
  --metric-mask data/data1/metric_mask.npy \
  --output data/data1/hierarchy.json
```

训练机房级模型：

```bash
python -m hiprotransfer.train_datacenter --config configs/default.yaml
```

训练服务级模型和原型：

```bash
python -m hiprotransfer.train_service --config configs/default.yaml
```

运行机器级检测：

```bash
python -m hiprotransfer.detect_machine --config configs/default.yaml
```

也可以一次性运行全部步骤：

```bash
sh scripts/run_pipeline.sh configs/default.yaml
```

检测阶段使用可配置的 sigma 阈值，默认是 3-sigma：

```text
threshold = mean(reference_scores) + sigma * std(reference_scores)
```

结果会写入：

```text
outputs/HiProTransfer/<dataset_name>/
```
