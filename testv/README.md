# 早停实验分析工具

## 环境要求

- Python 3.10+
- numpy

```bash
pip install numpy
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `compare.py` | **入口** — 运行这个 |
| `compare_early_stop.py` | 对比框架：双信号融合、分数配置解析、CSV 格式化 |
| `early_stop_hurst.py` | Hurst 指数早停：读 loss 序列，滑窗 DFA 计算 |
| `early_stop_hurst_cli.py` | Hurst 两阶段确认规则 |
| `early_stop_norm_snr.py` | 梯度范数 SNR 早停 |
| `jsonl_hurst.py` | DFA Hurst 滑窗计算引擎 |
| `metrics.py` | DFA Hurst 指数、Gini 系数等底层指标 |
| `smoothing.py` | Causal EWMA 平滑 |

## 使用方法

```bash
cd C:/Users/11236/Desktop/testv
python compare.py <包含_jsonl_的文件夹路径>
```

### 示例

```bash
python compare.py C:/Users/11236/Desktop/exported_5_proposals_20260729_181218
```

### 输出

运行后会在 `testv/` 目录下生成 **早停实验结果.csv**，包含以下列：

| 列名 | 说明 |
|---|---|
| 模型 | 自动从文件名提取（去掉 `logging_` 前缀） |
| Hurst标志步 | Hurst 指数首次触发 <= 阈值的步数 |
| SNR标志步 | 梯度信噪比首次触发 < 阈值的步数 |
| 截停步 | 两信号均触发后的最终截停步（取较晚者） |
| 融合loss | 首次 + 二次评估 loss 的加权融合（3:7） |
| 最终loss | 训练结束时的最佳 eval loss |
| 后续改善 | 截停后继续训练到结束的 loss 改善幅度 |

### JSONL 文件要求

每行一个 JSON，训练记录需包含 `loss` 和 `global_step/max_steps`（格式 `"当前步/总步"`），验证记录需包含 `eval_loss` 和 `eval_token_acc`。

文件名建议以 `logging_` 开头（如 `logging_v4.jsonl`），工具会自动去掉前缀作为模型名。

## 修改参数

编辑 `compare.py` 开头的两个配置字典：

```python
HURST_FRACTION_CONFIG = {
    "warmup_fraction": 0.10,      # 跳过前 10% 步
    "smoothing_fraction": 0.03,   # EWMA 平滑窗口比例
    "window_fraction": 0.18,      # Hurst 计算窗口比例
    "check_interval": 1,          # 检查间隔
    "threshold": 0.80,            # Hurst ≤ 0.80 触发
    "patience_fraction": 0.02,    # 连续触发耐心比例
    "observation_fraction": 0.04, # 两阶段间观察窗口比例
}

SNR_FRACTION_CONFIG = {
    "warmup_fraction": 0.10,
    "smoothing_fraction": 0.03,
    "window_fraction": 0.18,
    "reference_fraction": 0.11,   # 参考基线窗口比例
    "relative_threshold": 0.80,   # 相对基线 80% 以下触发
    "patience_fraction": 0.05,
}
```

所有 `_fraction` 参数都是相对于训练总步数的百分比，运行时会自动换算为实际记录数。
