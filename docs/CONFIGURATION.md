# 配置修改手册

## 1. 性能实验总配置

入口文件在 `configs/experiments/`。它本身不重复写所有参数，而是通过 `extends` 组合模型、硬件、AISBench、workload、搜索空间和验收规则。

以 `configs/experiments/qwen38-w4a8-speed2k.json` 为例：

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
  "name": "qwen38-w4a8-l28-35-speed2k"
}
```

要切模型、卡数、workload 或搜索规则，修改这里引用的文件即可。`name` 决定结果目录名。

## 2. 模型和 vLLM 基线参数

修改 `configs/models/*.json`。

| 字段 | 作用 |
|---|---|
| `model.path` | 宿主机模型目录 |
| `model.container_path` | Docker 内模型目录 |
| `model.served_name` | `/v1/models` 返回值以及客户端 model 名 |
| `model.tokenizer` | workload 长度核验使用的 tokenizer |
| `model.trust_remote_code` | 是否传 `--trust-remote-code` |
| `model.docker_args` | 模型专属 Docker 挂载；当前 W4A8 loader 补丁在这里 |
| `baseline.args` | vLLM 启动参数；键名对应 `--键名` |
| `baseline.env` | 传入 Docker 的环境变量 |

当前 W4A8 文件是 `configs/models/qwen38-w4a8-l28-35.json`。模型路径与 loader 补丁路径都在该文件中，换量化产物时必须同时检查二者。

布尔参数规则：

- `true` 生成 `--enable-...`；
- `false` 对 `enable-*` 生成 `--no-enable-...`；
- `null` 表示不向 vLLM 传该参数，使用后端默认值。

实验独有参数可以直接覆盖：

```json
"baseline": {
  "args": {
    "max-num-seqs": 8,
    "max-model-len": 131072,
    "max-num-batched-tokens": 16384,
    "gpu-memory-utilization": 0.9
  }
}
```

## 3. Docker、NPU、TP/DP 和端口

修改 `configs/hardware/ascend-single.json` 或 `ascend-dual.json`。

| 字段 | 作用 |
|---|---|
| `service.image` | vLLM Ascend Docker 镜像 |
| `service.port` | 自动启动服务监听端口 |
| `service.resource_ids` | 加锁用的物理资源名，防止两个任务占同一张卡 |
| `service.docker_args` | `--device`、驱动挂载、network、IPC、shm |
| `service.env.ASCEND_RT_VISIBLE_DEVICES` | 容器可见 NPU |
| `health_timeout_s` | 等待模型健康检查的最长时间 |
| `stop_timeout_s` | 优雅停止容器的等待时间 |
| `baseline.args.tensor-parallel-size` | TP 数 |
| `baseline.args.data-parallel-size` | DP 数 |

如果改用 NPU 2、3，至少同步修改 `resource_ids`、`--device /dev/davinciN` 和 `ASCEND_RT_VISIBLE_DEVICES`，三处必须一致。

## 4. AISBench 环境和超时

修改 `configs/benchmarks/aisbench.json`：

```json
{
  "benchmark": {
    "python": "/实际路径/envs/AISBench/bin/python",
    "binary": "/实际路径/envs/AISBench/bin/ais_bench",
    "timeout_s": 7200
  }
}
```

- `python`：控制器切换到的 Python。
- `binary`：AISBench 命令文件。
- `timeout_s`：单个 dataset/repeat 的最长时间，不是整个 search 的总时间。

修改后先运行 `doctor`，不要直接跑几个小时的任务。

## 5. 性能 workload

修改 `configs/workloads/*.json`。

| 字段 | 作用 |
|---|---|
| `concurrency` | 客户端同时提交的请求数，不等于 `max-num-seqs` |
| `input_tokens` | 每条目标输入 token；`null` 表示由数据本身决定 |
| `input_tolerance` | 实际输入与目标长度允许偏差 |
| `output_tokens` | 每条最大输出 token |
| `generation_kwargs` | temperature、ignore_eos 等生成参数 |
| `datasets` | 数据名、文件和文本字段 |
| `fast` | 搜索阶段 prompts/repeats/warmups |
| `final` | 最终固定测试 prompts/repeats/warmups |

100K/8 并发当前在 `configs/workloads/long100k-c8.json`；2K/32 并发在 `configs/workloads/speed2k-c32.json`。

## 6. 搜索空间

修改 `configs/search/*.json`。每个 phase 都从上一 phase 的胜者开始，只改本 phase 参数。

```json
{
  "name": "phase1",
  "strategy": "one_at_a_time",
  "parameters": {
    "args.max-num-batched-tokens": [8192, 16384, 32768]
  }
}
```

- 固定配置测试不读取候选值，不会迭代。
- `search --phase all` 才会依次跑 phase0 和所有启用 phase。
- `max_cases_per_phase` 是保险上限，避免搜索空间意外爆炸。

## 7. SLO 和排名

修改 `configs/selection/*.json`。

`constraints` 决定 `SLO_PASS/SLO_FAIL`，`objectives` 决定候选排序。当前 100K 验收以 `mean_decode_tps >= 15` 为生成速率条件，而不是最慢一路。

离线改门槛后可用 `report reclassify` 重判已有原始测量，不需要重跑模型。

## 8. 精度配置

精度配置独立放在 `configs/accuracy/*.json`，不经过性能实验的 `extends`。

| 字段 | 作用 |
|---|---|
| `endpoint.url` | OpenAI Chat Completions 地址 |
| `endpoint.model` | 必须与服务的 served model name 一致 |
| `endpoint.timeout_s` | 单题请求超时 |
| `dataset.file` | GSM8K JSONL 路径 |
| `limit` | 默认抽样数，`0` 表示全量 |
| `sample_seed` | 固定抽样和顺序 |
| `generation` | temperature、seed、max_tokens、thinking 设置 |
| `prompt_template` | 题目提示模板 |

BF16 与 W4A8 对比时必须保持 dataset、limit、sample_seed、generation 和 prompt_template 一致，仅修改 endpoint/model。比较命令会校验 workload hash，不一致会拒绝生成对比结论。

