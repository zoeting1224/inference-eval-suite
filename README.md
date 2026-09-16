# Inference Eval Suite

面向 vLLM/OpenAI 兼容服务的推理评测与自动寻优工具。项目只有一个命令入口：

```bash
python3 inferbench.py <功能> [参数]
```

它支持以下工作：

1. 打印并检查 Docker/vLLM 启动配置；
2. 自动启动 Docker/vLLM，按 phase 搜索性能参数；
3. 对已启动的服务运行固定配置性能测试；
4. 对已启动的服务运行 GSM8K 精度测试；
5. 把短文本拼接成精确 token 长度的长上下文数据；
6. 在不重跑模型的情况下刷新、重判或比较结果；
7. 用 tmux 在后台运行长任务并查看实时日志。

> 服务器部署目录：`/mnt/data1/zt/benchmark/inference_eval_suite`。以下命令默认从该目录执行。

## 1. 目录结构

```text
inference_eval_suite/
├── inferbench.py                 # 唯一公开入口
├── inference_suite/             # 实现代码；日常使用不需要修改
│   ├── runner.py                # Docker → vLLM → AISBench → 指标 → 选优
│   ├── service.py               # Docker/vLLM 启停和清理
│   ├── workload.py              # workload 冻结、AISBench 配置生成
│   ├── metrics.py               # 逐请求指标提取
│   ├── selection.py             # SLO 判断和候选排名
│   ├── external_perf.py         # 测试已经启动的服务
│   ├── accuracy.py              # GSM8K 精度测试
│   └── task.py                  # tmux 后台任务
├── configs/
│   ├── experiments/             # 性能实验总配置；引用下列配置
│   ├── accuracy/                # 精度测试的完整配置
│   ├── models/                  # 模型路径、served name、vLLM 通用参数
│   ├── hardware/                # Docker 镜像、NPU、TP/DP、端口和挂载
│   ├── benchmarks/              # AISBench 路径和超时
│   ├── workloads/               # 数据、并发、输入/输出长度、warmup/repeat
│   ├── search/                  # 搜索 phase、参数和候选值
│   └── selection/               # SLO 门槛和排名目标
├── data/
│   ├── performance/             # 性能数据和生成后的长上下文数据
│   └── accuracy/gsm8k/          # GSM8K 数据
├── runs/                        # 自动生成的所有结果
├── docs/                        # 补充说明
└── tests/                       # 自动化测试
```

## 2. 先选运行方式

| 目标 | 入口 | 模型由谁启动 |
|---|---|---|
| 只检查配置并查看完整 Docker/vLLM 命令 | `plan` | 不启动模型 |
| 检查模型、镜像、数据和 AISBench | `doctor` | 不启动模型 |
| 自动搜索最优参数 | `search` | 脚本逐 case 启停容器 |
| 用固定配置测试现有服务的性能 | `perf` | 用户提前启动 |
| 测试现有服务的 GSM8K 精度 | `accuracy` | 用户提前启动 |
| 构造 100K 等长上下文数据 | `data long-context` | 不启动模型 |
| 比较或重算已有结果 | `report` | 不请求模型 |
| 后台运行、看日志、停止任务 | `task` | 取决于后台命令 |

如果你只想复现某条客户启动命令，应手工启动服务后使用 `perf`，不要使用 `search`。如果需要脚本自动更换 vLLM 参数并逐组测试，使用 `search`。

## 3. 配置在哪里改

日常使用只修改 `configs/` 下的 JSON，不需要修改 `inference_suite/`。

| 要修改的内容 | 配置文件 | 主要字段 |
|---|---|---|
| 一次实验组合哪些配置 | `configs/experiments/*.json` | `extends`、`name`、`baseline.args` |
| 模型目录和模型名称 | `configs/models/*.json` | `model.path`、`container_path`、`served_name`、`tokenizer` |
| vLLM 基线参数和环境变量 | `configs/models/*.json` 或实验文件 | `baseline.args`、`baseline.env` |
| Docker 镜像和 NPU 数量 | `configs/hardware/*.json` | `service.image`、`resource_ids`、`docker_args`、`service.env` |
| TP、DP、服务端口 | `configs/hardware/*.json` | `tensor-parallel-size`、`data-parallel-size`、`service.port` |
| AISBench Conda 环境和超时 | `configs/benchmarks/aisbench.json` | `benchmark.python`、`binary`、`timeout_s` |
| 数据、并发和 token 长度 | `configs/workloads/*.json` | `datasets`、`concurrency`、`input_tokens`、`output_tokens` |
| warmup、重复次数和请求数 | `configs/workloads/*.json` | `fast`、`final` |
| 自动搜索哪些参数 | `configs/search/*.json` | `search.phases[].parameters` |
| 达标条件和排名方式 | `configs/selection/*.json` | `constraints`、`objectives` |
| 精度测试服务和抽样数 | `configs/accuracy/*.json` | `endpoint`、`dataset`、`limit`、`generation` |

### 3.1 性能实验总配置

实验文件把模型、硬件、AISBench、workload、搜索空间和 SLO 组合起来：

```json
{
  "extends": [
    "../models/qwen38-w4a8-l28-35.json",
    "../hardware/ascend-dual.json",
    "../benchmarks/aisbench.json",
    "../workloads/speed2k-c32.json",
    "../search/general.json",
    "../selection/throughput.json"
  ],
  "name": "qwen38-w4a8-l28-35-speed2k",
  "baseline": {
    "args": {
      "max-num-seqs": 32,
      "max-model-len": 4096,
      "max-num-batched-tokens": 16384,
      "gpu-memory-utilization": 0.85
    }
  }
}
```

`extends` 从上到下合并，后面的值覆盖前面的值，实验文件自身最后覆盖。因此某个实验独有的 vLLM 参数应写在实验文件的 `baseline.args` 中。

### 3.2 换模型

复制一个 `configs/models/*.json`，至少修改：

```json
{
  "model": {
    "path": "/宿主机/模型目录",
    "container_path": "/workspace/model",
    "served_name": "API中的模型名",
    "tokenizer": "/宿主机/模型目录"
  },
  "baseline": {
    "args": {
      "quantization": "ascend"
    }
  }
}
```

不量化模型可参考 `configs/models/example-unquantized.json`。模型额外文件需要映射进容器时，将参数加入 `model.docker_args`。

### 3.3 改 Docker、卡数和 TP/DP

- 单卡模板：`configs/hardware/ascend-single.json`
- 双卡模板：`configs/hardware/ascend-dual.json`

重点修改：

```json
{
  "service": {
    "image": "你的vLLM镜像",
    "port": 8004,
    "resource_ids": ["ascend-0", "ascend-1"],
    "env": {"ASCEND_RT_VISIBLE_DEVICES": "0,1"}
  },
  "baseline": {
    "args": {
      "tensor-parallel-size": 2,
      "data-parallel-size": 1
    }
  }
}
```

同时确认 `service.docker_args` 中的 `/dev/davinciN` 与实际卡号一致。

### 3.4 改性能 workload

例如 `configs/workloads/long100k-c8.json`：

```json
{
  "workload": {
    "concurrency": 8,
    "input_tokens": 100000,
    "output_tokens": 512,
    "generation_kwargs": {"temperature": 0.0, "ignore_eos": true},
    "datasets": [
      {"name": "long_context", "path": "data/performance/long100k_c8/requests.jsonl", "text_field": "question"}
    ],
    "fast": {"prompts_per_dataset": 8, "repeats": 1, "warmups": 3},
    "final": {"prompts_per_dataset": 8, "repeats": 3, "warmups": 3}
  }
}
```

- `max-num-seqs` 是服务容量上限；测试实际并发由这里的 `workload.concurrency` 决定。
- `input_tokens: null` 表示使用数据集本身长度。
- `output_tokens` 是每条请求的输出上限；配合 `ignore_eos: true` 会尽量生成满。
- `fast` 用于搜索，`final` 用于最终复测。

### 3.5 改搜索空间与 SLO

在 `configs/search/*.json` 中配置候选值，例如：

```json
{"args.max-num-batched-tokens": [8192, 16384, 32768]}
```

在 `configs/selection/*.json` 中配置硬门槛和排序目标，例如：

```json
{
  "constraints": {"failed_requests": {"max": 0}},
  "objectives": [
    {"metric": "output_tps", "direction": "max"},
    {"metric": "tpot_ms", "direction": "min"}
  ]
}
```

## 4. 首次使用：两步检查

```bash
cd /mnt/data1/zt/benchmark/inference_eval_suite

# 1. 只解析配置并打印完整 Docker/vLLM 命令
python3 inferbench.py plan \
  --config configs/experiments/qwen38-w4a8-speed2k.json

# 2. 检查模型、tokenizer、Docker 镜像、数据和 AISBench
python3 inferbench.py doctor \
  --config configs/experiments/qwen38-w4a8-speed2k.json
```

性能测试和自动搜索依赖 AISBench。先把 `configs/benchmarks/aisbench.json` 中的 `python`、`binary` 和 `timeout_s` 改成服务器真实值。精度测试不依赖 AISBench。

## 5. 功能一：固定配置性能测试

适合测试手工启动的服务或复现客户启动配置。`perf` 不创建、重启或停止 Docker。

```bash
# 先确认服务和 served model name
curl -s http://127.0.0.1:8000/v1/models

# 对现有 8000 端口运行 final workload
python3 inferbench.py perf \
  --config configs/experiments/qwen38-w4a8-speed2k.json \
  --mode final \
  --port 8000 \
  --output runs/performance/w4a8-fixed-001
```

`--port` 只覆盖服务端口；数据、并发、输出长度、warmup 和 repeat 来自实验引用的 workload 配置。每次使用新的 `--output`，避免混合不同测试。

## 6. 功能二：自动寻优

先执行 `plan` 和 `doctor`，确认无误后放到 tmux 后台：

```bash
python3 inferbench.py task start --name w4a8-search -- \
  search \
  --config configs/experiments/qwen38-w4a8-speed2k.json \
  --phase all
```

`--phase all` 会运行 `phase0` 基线和搜索配置中启用的全部 phase。每个 case 的流程为：

```text
生成候选配置 → 启动独立 Docker/vLLM → 健康检查
→ AISBench → 指标/SLO/排名 → 停止本 case 容器 → 继承最优配置
```

同一实验身份和 phase 已完整完成时，重复执行会返回已有结果，不浪费时间。确实需要重新测试时加 `--new-run`。

### 最终 baseline/best 公平对比

搜索完成后，使用完全相同的 `final` workload 复测原始 baseline 和最优配置：

```bash
python3 inferbench.py search \
  --config configs/experiments/qwen38-w4a8-speed2k.json \
  --phase final-baseline

python3 inferbench.py search \
  --config configs/experiments/qwen38-w4a8-speed2k.json \
  --phase final
```

`final-baseline` 读取原始 `baseline`，`final` 读取搜索保存的 `best_candidate.json`。两边 workload hash 一致后会生成 A/B comparison 报告。

## 7. 功能三：固定配置精度测试

精度测试使用已启动的 OpenAI 兼容服务。先跑 1 题冒烟，再启动正式任务：

```bash
python3 inferbench.py accuracy \
  --config configs/accuracy/qwen38-w4a8-l28-35-gsm8k.json \
  --limit 1 \
  --output runs/accuracy/w4a8-smoke

python3 inferbench.py task start --name w4a8-gsm8k -- \
  accuracy \
  --config configs/accuracy/qwen38-w4a8-l28-35-gsm8k.json \
  --output runs/accuracy/w4a8-gsm8k-200
```

服务地址和模型名在精度配置的 `endpoint.url`、`endpoint.model` 中修改；题数由 `limit` 控制。相同 `--output` 中断续跑时会跳过 `predictions.jsonl` 中已完成的题目。

比较 BF16 和量化模型精度：

```bash
python3 inferbench.py report accuracy-compare \
  --baseline runs/accuracy/bf16-gsm8k-200 \
  --candidate runs/accuracy/w4a8-gsm8k-200 \
  --output runs/accuracy/bf16-vs-w4a8
```

## 8. 功能四：生成 100K 长上下文数据

```bash
python3 inferbench.py data long-context \
  --config configs/experiments/qwen38-bf16-long100k.json \
  --source data/performance/throughput_2k_high_entropy.jsonl \
  --source data/performance/throughput_2k_low_entropy.jsonl
```

目标 token 数和输出路径来自实验引用的 workload。命令会生成 `requests.jsonl` 和记录来源、token 数、seed、哈希的 manifest。已有目标默认不覆盖；明确需要重建时加 `--force`。

原始 JSONL 是本地测试资产，不提交到公开仓库。需要放置的路径见 [data/README.md](data/README.md)。

## 9. 功能五：后台任务和实时日志

```bash
# 查看所有本项目后台任务
python3 inferbench.py task status

# 查看最近 300 行输出
python3 inferbench.py task logs --name w4a8-search --lines 300

# 进入 tmux 实时查看
python3 inferbench.py task attach --name w4a8-search

# 停止任务；搜索任务会清理它自己创建的容器
python3 inferbench.py task stop --name w4a8-search
```

tmux session 会自动添加 `inferbench-` 前缀。已经处于 tmux 中时优先使用 `task logs`，避免嵌套 attach。

## 10. 功能六：不重跑模型处理报告

```bash
# 根据已有原始结果重建报告
python3 inferbench.py report refresh --session <完整session目录>

# 修改 SLO 后重判已有结果
python3 inferbench.py report reclassify \
  --config configs/experiments/qwen38-bf16-long100k.json \
  --root <session或上级目录>

# 手工指定两次 final 结果生成 A/B 对比
python3 inferbench.py report compare \
  --baseline <final-baseline-session> \
  --candidate <final-session>
```

## 11. 结果从哪里看

所有结果只写入 `runs/`：

- 性能测试先看 `summary.md`、`ranking.csv` 和 `slo_report.json`；
- 逐请求成功状态、TTFT、TPOT、decode TPS 看 `requests.csv`；
- 流程异常先看 session 的 `master.log`；
- 模型启动或运行异常看 case 下的 `vllm.log`；
- AISBench 超时或客户端错误看 `aisbench.log`；
- 精度结果先看 `summary.md`，逐题结果看 `predictions.jsonl`；
- `workload_profile.json` 记录并发、请求数、warmup、repeat 和 workload hash。

## 12. 当前示例配置

| 配置 | 用途 |
|---|---|
| `configs/experiments/qwen38-w4a8-speed2k.json` | W4A8、双卡、2K 性能测试/搜索 |
| `configs/experiments/qwen38-bf16-long100k.json` | BF16、双卡、100K 输入性能测试/搜索 |
| `configs/accuracy/qwen38-w4a8-l28-35-gsm8k.json` | W4A8 GSM8K 精度测试 |
| `configs/accuracy/qwen38-bf16-gsm8k.json` | BF16 GSM8K 对照测试 |

更多边界说明见 [配置说明](docs/CONFIGURATION.md)、[运行流程](docs/WORKFLOWS.md) 和 [结果文件说明](docs/OUTPUTS.md)。
