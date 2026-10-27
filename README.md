# Inference Eval Suite

面向 vLLM/OpenAI 兼容服务的推理评测与自动寻优工具。项目只有一个命令入口：

```bash
python3 inferbench.py <功能> [参数]
```

这里的 `<功能>` 和 `[参数]` 是占位符，不要原样输入：

- `<功能>`：紧跟在 `inferbench.py` 后面，替换为 `plan`、`doctor`、`search`、`perf`、`accuracy`、`data`、`report` 或 `task`。
- `[参数]`：写在功能后面，例如配置文件、phase、端口和输出目录。

命令的实际结构是：

```text
python3 inferbench.py  功能名  该功能所需的参数
                      ↑       ↑
                    在这里选   在这里填写配置、端口、输出目录等
```

例如选择“固定配置性能测试”时，功能名是 `perf`：

```bash
cd /mnt/data1/zt/benchmark/inference_eval_suite

python3 inferbench.py perf \
  --config configs/experiments/qwen38-w4a8-speed2k.json \
  --mode final \
  --port 8000 \
  --output runs/performance/w4a8-fixed-001
```

也就是说，不需要在代码文件中“开启”某个功能；运行命令时，把选中的功能名放在 `inferbench.py` 后面即可。

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

选好后，把“功能入口”一列的内容直接放在 `python3 inferbench.py` 后面。下面每行都给出了一条可以照着修改的完整示例。

| 目标 | 功能入口 | 完整示例 | 模型由谁启动 |
|---|---|---|---|
| 只检查配置并查看完整 Docker/vLLM 命令 | `plan` | `python3 inferbench.py plan --config configs/experiments/qwen38-w4a8-speed2k.json` | 不启动模型 |
| 检查模型、镜像、数据和 AISBench | `doctor` | `python3 inferbench.py doctor --config configs/experiments/qwen38-w4a8-speed2k.json` | 不启动模型 |
| 自动搜索最优参数（全面预设） | `search` | `python3 inferbench.py search --config configs/experiments/qwen38-w8a8-comprehensive.json --phase all` | 脚本逐 case 启停容器 |
| 用固定配置测试现有服务的性能 | `perf` | `python3 inferbench.py perf --config configs/experiments/qwen38-w4a8-speed2k.json --mode final --port 8000 --output runs/performance/test-001` | 用户提前启动 |
| 测试现有服务的 GSM8K 精度 | `accuracy` | `python3 inferbench.py accuracy --config configs/accuracy/qwen38-w4a8-l28-35-gsm8k.json --limit 1 --output runs/accuracy/smoke` | 用户提前启动 |
| 构造 100K 等长上下文数据 | `data long-context` | `python3 inferbench.py data long-context --config configs/experiments/qwen38-bf16-long100k.json --source data/performance/throughput_2k_high_entropy.jsonl` | 不启动模型 |
| 比较两次性能结果 | `report compare` | `python3 inferbench.py report compare --baseline <baseline目录> --candidate <candidate目录>` | 不请求模型 |
| 在 tmux 后台启动任务 | `task start` | `python3 inferbench.py task start --name w8a8-full-search -- search --config configs/experiments/qwen38-w8a8-comprehensive.json --phase all` | 取决于后台命令 |

例如你选择“自动搜索最优参数”，就执行表中的 `search` 示例；选择“已经启动模型，只跑一次固定性能测试”，就执行 `perf` 示例。

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

搜索空间不只是若干候选数字，还定义了调参顺序、组合方式以及上一阶段最优配置如何传给下一阶段。推荐按收益优先级分阶段搜索，避免一次性做全部参数的全排列。

完整的 Qwen/vLLM 搜索设计如下：

| 阶段 | 优先级 | 参数 | 推荐候选 | 组合方式 |
|---|---:|---|---|---|
| `phase0` | 基线 | 原始 `baseline` | 1 个基线 | 固定配置 |
| `phase1` | 高 | MTP speculative tokens | 关闭 / 1 / 2 / 3 / 4 | 单参数扫描 |
| `phase2` | 高 | `max-num-seqs` × `max-num-batched-tokens` | 16 / 32 / 64 × 16384 / 32768 | 小型网格 |
| `phase3` | 中 | `gpu-memory-utilization` | 0.90 / 0.95 | 单参数扫描 |
| `phase3` | 中 | Prefix Cache | on / off | 单参数扫描 |
| `phase3` | 中 | `max-model-len` | 131072 / 262144 | 单参数扫描 |
| `phase4` | 中 | Graph 模式 | `FULL_DECODE_ONLY` / 默认 / eager | 单参数扫描 |
| `phase5` | 低 | HCCL、OMP、CPU binding | 256/512/1024；1/2/4/8；on/off | 最后手动精调 |

搜索阶段按顺序继承 incumbent：`phase1` 选出的最佳 MTP 会成为 `phase2` 的起点，随后继续搜索 Batch、显存、Graph 和系统参数。不要把所有维度直接做笛卡尔积，否则 case 数量会迅速失控。

新版已内置可直接引用的 `configs/search/qwen-comprehensive.json`，内容如下。它是旧版 `search_space.env` 的可执行 JSON 迁移版，不需要用户重新手写全部候选值；MTP 的 `method` 仍必须按模型和 vLLM 后端核对：

```json
{
  "search": {
    "max_cases_per_phase": 12,
    "phases": [
      {
        "name": "phase1",
        "description": "Qwen MTP",
        "strategy": "one_at_a_time",
        "parameters": {
          "args.speculative-config": [
            null,
            {"method": "qwen3_5_mtp", "num_speculative_tokens": 1, "enforce_eager": true},
            {"method": "qwen3_5_mtp", "num_speculative_tokens": 2, "enforce_eager": true},
            {"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true},
            {"method": "qwen3_5_mtp", "num_speculative_tokens": 4, "enforce_eager": true}
          ]
        }
      },
      {
        "name": "phase2",
        "description": "Batch scheduler",
        "strategy": "grid",
        "skip_incompatible": true,
        "parameters": {
          "args.max-num-seqs": [16, 32, 64],
          "args.max-num-batched-tokens": [16384, 32768]
        }
      },
      {
        "name": "phase3",
        "description": "KV cache and context budget",
        "strategy": "one_at_a_time",
        "skip_incompatible": true,
        "parameters": {
          "args.gpu-memory-utilization": [0.9, 0.95],
          "args.enable-prefix-caching": [true, false],
          "args.max-model-len": [131072, 262144]
        }
      },
      {
        "name": "phase4",
        "description": "Graph mode",
        "strategy": "variants",
        "variants": [
          {
            "args.compilation-config": {"cudagraph_mode": "FULL_DECODE_ONLY"},
            "args.enforce-eager": null
          },
          {
            "args.compilation-config": null,
            "args.enforce-eager": null
          },
          {
            "args.compilation-config": null,
            "args.enforce-eager": true
          }
        ]
      },
      {
        "name": "phase5",
        "description": "Low-priority communication and CPU tuning",
        "strategy": "one_at_a_time",
        "enabled": false,
        "parameters": {
          "env.HCCL_BUFFSIZE": ["256", "512", "1024"],
          "env.OMP_NUM_THREADS": ["1", "2", "4", "8"],
          "args.additional-config": [
            {"enable_cpu_binding": true},
            {"enable_cpu_binding": false}
          ]
        }
      }
    ]
  }
}
```

可直接运行的 Qwen3.8 W8A8 组合入口是 `configs/experiments/qwen38-w8a8-comprehensive.json`。其他模型也可以在自己的实验配置中引用该搜索文件，但运行前仍要核对：

- `max-num-seqs` 不能小于 `workload.concurrency`；`skip_incompatible=true` 会自动跳过这类候选并记录原因。
- `max-model-len` 必须覆盖模板后的输入 token 与输出 token 之和；不兼容候选会自动跳过，但 2K 测试仍建议主动删除没有意义的 131K/262K 候选以节省时间。
- `cache_policy=cold` 时 Prefix Cache 必须为 off；预设会跳过 on，只有 warm workload 才真正比较 on/off。
- `phase5` 收益通常较低，默认 `enabled=false`，在高收益阶段稳定后再单独启用。
- Graph 使用 `variants`，确保 `FULL_DECODE_ONLY`、后端默认和 `--enforce-eager` 三种模式互斥，不会拼出无效组合。
- `plan` 会列出每个 phase 的 runnable/skipped 数量和跳过原因；正式搜索目录还会保存 `phaseN_skipped.json`。

旧项目中的全面搜索并非只依赖 `search_space.env`。该文件只保存候选数组；`tune.sh` 还负责 phase 顺序、网格组合和最佳配置继承。新版已把这些行为统一放入执行器：候选值在 `qwen-comprehensive.json`，组合/继承/兼容性检查由 `inference_suite/` 完成。旧 `.env` 不再作为运行入口。

只查看全面搜索将运行哪些 case，不启动模型：

```bash
python3 inferbench.py plan \
  --config configs/experiments/qwen38-w8a8-comprehensive.json
```

如需最后再调低收益 phase5（它不会包含在默认 `all` 中）：

```bash
python3 inferbench.py search \
  --config configs/experiments/qwen38-w8a8-comprehensive.json \
  --phase phase5
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

自动寻优推荐使用上一节的完整分阶段设计，而不是只搜索 Batch Tokens 和显存比例。默认顺序为：

```text
phase0 baseline
  → phase1 MTP
  → phase2 max-num-seqs × max-num-batched-tokens
  → phase3 KV Cache / Prefix Cache / max-model-len
  → phase4 Graph
  → phase5 HCCL / OMP / CPU binding（低优先级，默认关闭）
  → final-baseline 与 final 公平复测
```

每个 phase 只修改本组参数，其余参数继承当前最佳配置。`phase2` 采用小型网格；其他阶段优先使用 `one_at_a_time`，以控制总 case 数。

先执行 `plan` 和 `doctor`，确认无误后放到 tmux 后台：

```bash
python3 inferbench.py task start --name w8a8-full-search -- \
  search \
  --config configs/experiments/qwen38-w8a8-comprehensive.json \
  --phase all
```

`--phase all` 会运行 `phase0` 基线和搜索配置中启用的全部 phase。每个 case 的流程为：

```text
生成候选配置 → 启动独立 Docker/vLLM → 健康检查
→ AISBench → 指标/SLO/排名 → 停止本 case 容器 → 继承最优配置
```

同一实验身份和 phase 已完整完成时，重复执行会返回已有结果，不浪费时间。确实需要重新测试时加 `--new-run`。

运行前重点检查 `plan` 输出：实际 workload 并发、每个 phase 的 runnable/skipped case 数、MTP 方法、`max-model-len`、Prefix Cache 策略以及 Docker/NPU 映射。低收益 `phase5` 默认关闭，不会被第一次 `all` 执行。

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

判分只接受模型明确给出的 `#### <number>` 或 `\\boxed{<number>}` 最终答案；不再把未完成推理过程中的最后一个数字当成答案。`summary.json` 会分别记录缺少最终答案的题数、达到输出上限的题数，以及两者重叠的题数。`predictions.jsonl` 保留每题的 `finish_reason` 和输出 token 用量。改变 `generation.max_tokens` 或判分规则后，必须使用新的 `--output` 目录；脚本会拒绝在旧目录上混合续跑。

例如复测 1024-token 输出上限，不需要改动历史配置文件：

```bash
python3 inferbench.py accuracy \
  --config configs/accuracy/qwen38-w4a8-l8-55-gsm8k.json \
  --limit 200 \
  --max-tokens 1024 \
  --output runs/accuracy/qwen38-w4a8-l8-55-gsm8k-200-out1024
```

`--max-tokens` 的实际取值会写入该次结果的 `manifest.json` 和 `summary.json`。四组对照都要使用相同数值及各自的新目录。

比较 BF16 和量化模型精度：

```bash
python3 inferbench.py report accuracy-compare \
  --baseline runs/accuracy/bf16-gsm8k-200 \
  --candidate runs/accuracy/w4a8-gsm8k-200 \
  --output runs/accuracy/bf16-vs-w4a8
```

命令会同时输出：

- `comparison.md/json`：准确率、Both correct、Both wrong、两类单边正确数量和 W4A8 净增益；
- `disagreements.md`：BF16 正确/W4A8 错误以及 BF16 错误/W4A8 正确的题目、Gold 和两边预测；
- `disagreements.jsonl`：便于后续程序分析的逐题分歧记录。

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
