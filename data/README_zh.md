# 数据目录

## 数据开源范围

受公司保密要求、生产系统安全要求和数据治理规定限制，完整生产环境数据集无法公开。因此，本仓库仅发布 `data1`：从全量数据中剪枝得到、经过匿名化处理且保留层次拓扑关系的代表性子集。

该子集包含 3 个机房、6 个服务和 50 台机器，主要用于展示 HiProTransfer 的“机房 -> 服务 -> 机器”三级结构，并支持开源流水线的功能复现。

## 已包含文件

```text
data/data1/
  offline_data.npy
  online_data.npy
  labels.npy
  machine_list.json
  metric_mask.npy
  hierarchy.json
```

拓扑分布：

| 机房 | 服务 | 机器数 |
|---|---|---:|
| datacenters_1 | server_1 | 12 |
| datacenters_1 | server_2 | 6 |
| datacenters_2 | server_3 | 12 |
| datacenters_2 | server_4 | 8 |
| datacenters_3 | server_2 | 10 |
| datacenters_3 | server_5 | 2 |

## 预处理与文件说明

本目录保存的是**预处理之后的数据**，不是原始生产指标。离线数据和在线数据均已按照原实验流程，基于每台机器的离线数据完成 MinMax 预处理。

由于 NumPy 数组需要使用规则的矩形形状，而不同机器实际拥有的指标集合可能不同，为了构造统一的 19 维指标张量，我们将不可用的缺失指标填充为 `0`。`metric_mask.npy` 会明确记录每台机器的每个指标是否真实存在，从而区分“缺失指标使用 0 占位”和“指标真实观测值本身就是 0”这两种情况。机器、机房、服务和指标标识均已匿名化。

### 索引对齐关系

所有数组的第一维使用同一套公开机器索引：

```text
machine_list[i]
offline_data[i, :, :]
online_data[i, :, :]
labels[i, :]
metric_mask[i, :]
```

`hierarchy.json` 中出现的机器索引也使用这套从 0 开始的索引。公开索引是针对本子集重新编号的，不是全量私有数据中原有的机器索引。

### `offline_data.npy`

- 形状：`[50, 20160, 19]`
- 类型：`float32`
- 各维含义：`[机器, 时间点, 指标]`
- 时间范围：每台机器包含 20,160 个一分钟粒度时间点，即连续 14 个完整天。
- 数值含义：每台机器独立使用其离线序列拟合预处理参数；有效且非恒定的指标被缩放到 `[-1, 1]`。
- 用途：构建层次结构，以及机房级、服务级和机器级模型训练。

### `online_data.npy`

- 形状：`[50, 10080, 19]`
- 类型：`float32`
- 各维含义：`[机器, 时间点, 指标]`
- 时间范围：每台机器包含 10,080 个一分钟粒度时间点，即连续 7 个完整天。
- 数值含义：已使用对应机器离线数据拟合得到的预处理参数完成转换。超出离线取值范围的数据可能超过 `[-1, 1]`；按照原预处理流程，公开的在线数据最终被截断到闭区间 `[-3, 3]`。
- 用途：异常检测和结果评估。

### `labels.npy`

- 形状：`[50, 10080]`
- 类型：`int32`
- 各维含义：`[机器, 在线时间点]`
- 标签语义：`0` 表示正常，`1` 表示异常。
- 对齐方式：`labels[i, t]` 是 `online_data[i, t, :]` 对应时间点的标签。
- 离线数据没有标签文件，因为离线段作为无监督训练和参考数据使用。

### `machine_list.json`

一个包含 50 个匿名机器名的有序 JSON 数组。数组位置 `i` 就是其他所有文件使用的公开机器索引。

机器名格式为：

```text
datacenters_<机房>-server_<服务>-machine_<机器>
```

例如，`datacenters_1-server_1-machine_1` 属于机房 `datacenters_1` 和服务 `server_1`。机房编号连续为 1–3，服务编号连续为 1–5，机器编号连续为 1–50。这些标识均为匿名编号，不对应公开的生产环境真实名称。

机器名中的 `machine_<n>` 是从 1 开始的公开编号，而 NumPy 数组仍使用从 0 开始的下标。因此，`machine_1` 位于数组下标 `0`，`machine_2` 位于数组下标 `1`，依此类推。

### `metric_mask.npy`

- 形状：`[50, 19]`
- 类型：`uint8`
- 语义：`metric_mask[i, j] == 1` 表示机器 `i` 存在第 `j` 个指标；值为 `0` 表示该指标不可用，对应的数据列以零填充。
- 使用要求：训练和评分时必须通过该 mask 排除不可用维度，不能把缺失指标的零填充值当作真实观测。当 mask 为 `1` 时，数据中的 `0` 是合法的预处理后观测值，不表示指标缺失。
- 指标位置使用匿名的 `0`–`18` 下标，与离线和在线数组的最后一维严格对应。

### `hierarchy.json`

该文件编码以下三级拓扑：

```text
机房 -> 服务 -> 机器
```

结构示例：

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

字段说明：

- `datacenters`：以匿名机房 ID 为键的映射。
- `services`：某机房下以匿名服务 ID 为键的映射。
- `machines`：公开机器索引列表，从 0 开始，可直接索引所有 `.npy` 数组和 `machine_list.json`。
- `machine_names`：与 `machines` 逐位置对应的匿名机器名。
- 机器名中的数字后缀从 1 开始，而 `machines` 和 `center_machine` 中保存的是从 0 开始的数组下标。
- `center_machine`：该服务中心机器的公开索引。代码会计算每台候选机器与同一服务内其他机器的平均距离，计算距离时只使用两台机器共同存在的指标；平均距离最小的机器被选为该服务的中心机器。该字段不是服务内列表的相对位置，也不是根据异常检测效果选择的。
- 服务级 `machine_count`：该服务包含的机器数。
- 机房级 `machine_count`：该机房所有服务的机器数之和。

以下关系始终成立：

```text
machine_names[k] == machine_list[machines[k]]
```

## 数据加载示例

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
print(metric_mask[machine_idx])       # 19 维指标有效性 mask
```

## 运行

在开源仓库根目录运行：

```bash
sh scripts/run_pipeline.sh configs/default.yaml data/data1
```
