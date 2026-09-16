# Inference Eval Suite

这是从 `qwen38_w8a8_tuning` 拆出的干净副本。新项目只保留一条公开命令入口：

```bash
python3 inferbench.py <功能> [参数]
```

原项目 `/mnt/data1/zt/benchmark/qwen38_w8a8_tuning` 不会被修改；新项目不包含旧 Shell 流程、旧结果、旧状态、临时补丁或大日志。

## 目录结构

```text
inference_eval_suite/
├── inferbench.py                 # 唯一公开入口
├── inference_suite/             # 实现代码；日常改配置不需要修改
│   ├── cli.py                   # 命令路由
│   ├── session.py               # 搜索会话、断点续跑、防重复执行
│   ├── runner.py                # Docker → vLLM → AISBench → 指标 → 选优
│   ├── service.py               # Docker/vLLM 启停及只清理自有容器
│   ├── workload.py              # 冻结 workload、生成 AISBench 配置
│   ├── metrics.py               # 逐请求指标提取
│   ├── selection.py             # SLO 判断及排名
│   ├── external_perf.py         # 对已启动服务做固定性能测试
│   ├── accuracy.py              # 对已启动服务做 GSM8K 精度测试
│   ├── long_context.py          # 构造精确 token 长度的数据
│   ├── reporting.py             # summary/ranking 报告
│   ├── reclassify.py            # 不重跑，按新 SLO 重判历史结果
│   └── task.py                  # tmux 后台任务管理
├── configs/
│   ├── experiments/             # 性能测试总配置；组合下列配置
│   ├── accuracy/                # 固定精度测试配置
│   ├── models/                  # 模型路径、served name、vLLM 基线参数
│   ├── hardware/                # Docker 镜像、NPU、TP/DP、端口、挂载
│   ├── benchmarks/              # AISBench Python、二进制、超时
│   ├── workloads/               # 数据、输入/输出长度、并发、warmup/repeat
│   ├── search/                  # 搜索 phase 和候选参数
│   └── selection/               # SLO 门槛和排序目标
├── data/
│   ├── performance/             # 性能测试源数据和生成后的长上下文数据
│   └── accuracy/gsm8k/          # GSM8K 固定测试集
├── runs/                        # 自动生成；所有结果只写这里
├── docs/
│   ├── CONFIGURATION.md         # 哪个配置改什么
│   ├── WORKFLOWS.md             # 每个功能怎样运行
│   └── OUTPUTS.md               # 结果文件怎样看
└── tests/                       # 自动化测试
```

## 功能与唯一入口

| 功能 | 命令入口 | 是否启动/停止模型 |
|---|---|---|
| 查看计划和完整 Docker 命令 | `inferbench.py plan` | 否 |
| 环境、镜像、模型、数据自检 | `inferbench.py doctor` | 否 |
| 自动寻优 | `inferbench.py search` | 是，只管理本次创建的容器 |
| 固定配置性能测试 | `inferbench.py perf` | 否，使用现有服务 |
| 固定配置精度测试 | `inferbench.py accuracy` | 否，使用现有服务 |
| 构造长上下文数据 | `inferbench.py data long-context` | 否 |
| 比较、刷新、重判报告 | `inferbench.py report ...` | 否，不重新请求模型 |
| tmux 后台运行和看日志 | `inferbench.py task ...` | 取决于后台命令 |

## 当前可直接使用的配置

- `configs/accuracy/qwen38-w4a8-l28-35-gsm8k.json`：当前 W4A8 服务的 GSM8K 精度测试。
- `configs/accuracy/qwen38-bf16-gsm8k.json`：BF16 对照精度测试。
- `configs/experiments/qwen38-w4a8-speed2k.json`：W4A8、双卡、2K 性能测试/搜索。
- `configs/experiments/qwen38-bf16-long100k.json`：BF16、双卡、100K 输入性能测试/搜索。

## 首次使用

```bash
cd /mnt/data1/zt/benchmark/inference_eval_suite

# 仅解析配置并打印命令，不启动容器
python3 inferbench.py plan \
  --config configs/experiments/qwen38-w4a8-speed2k.json

# 检查 AISBench、Docker 镜像、模型和数据是否就绪
python3 inferbench.py doctor \
  --config configs/experiments/qwen38-w4a8-speed2k.json
```

当前服务器尚未发现 AISBench 环境。性能测试和自动寻优前，必须在 `configs/benchmarks/aisbench.json` 填写真实的 `python` 与 `binary` 路径。精度测试不依赖 AISBench，可以直接对现有 OpenAI 兼容服务运行。

原始 JSONL 数据属于本地测试资产，不提交到公开仓库；放置位置见 [data/README.md](data/README.md)。服务器现有数据不受影响。

详细配置见 [docs/CONFIGURATION.md](docs/CONFIGURATION.md)，完整运行步骤见 [docs/WORKFLOWS.md](docs/WORKFLOWS.md)。
