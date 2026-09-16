# 运行流程

以下命令都从项目根目录运行：

```bash
cd /mnt/data1/zt/benchmark/inference_eval_suite
```

## 1. 查看配置和模型启动命令

```bash
python3 inferbench.py plan \
  --config configs/experiments/qwen38-w4a8-speed2k.json
```

它只验证配置并打印 Docker/vLLM 命令，不启动容器、不占用 NPU、不产生测试结果。改参数后应先跑一次。

## 2. 运行环境自检

```bash
python3 inferbench.py doctor \
  --config configs/experiments/qwen38-w4a8-speed2k.json
```

它检查模型、tokenizer、数据、Docker 镜像、AISBench 路径和 workload hash。全部通过后再运行性能任务。

## 3. 对已经启动的服务做固定精度测试

先确认服务模型名：

```bash
curl -s http://127.0.0.1:8000/v1/models
```

W4A8 先跑 1 题冒烟：

```bash
python3 inferbench.py accuracy \
  --config configs/accuracy/qwen38-w4a8-l28-35-gsm8k.json \
  --limit 1 \
  --output runs/accuracy/w4a8-smoke
```

再跑配置默认的 200 题：

```bash
python3 inferbench.py task start --name w4a8-gsm8k -- \
  accuracy \
  --config configs/accuracy/qwen38-w4a8-l28-35-gsm8k.json \
  --output runs/accuracy/w4a8-gsm8k-200
```

中断后用相同 `--output` 再运行会读取 `predictions.jsonl` 并续跑，已完成题目不会重复请求。

BF16 使用：

```bash
python3 inferbench.py accuracy \
  --config configs/accuracy/qwen38-bf16-gsm8k.json \
  --output runs/accuracy/bf16-gsm8k-200
```

两次测试结束后比较：

```bash
python3 inferbench.py report accuracy-compare \
  --baseline runs/accuracy/bf16-gsm8k-200 \
  --candidate runs/accuracy/w4a8-gsm8k-200 \
  --output runs/accuracy/bf16-vs-w4a8
```

## 4. 对已经启动的服务做固定性能测试

`perf` 不会创建或停止 Docker，适合复现客户启动命令或测试手工启动的服务。

```bash
python3 inferbench.py perf \
  --config configs/experiments/qwen38-w4a8-speed2k.json \
  --mode final \
  --port 8000 \
  --output runs/performance/w4a8-fixed-001
```

这里的数据文件、并发、输出长度、warmup 和 repeat 都来自实验引用的 workload 配置。`--port` 只覆盖服务端口。输出目录必须是新目录，防止把不同测试混在一起。

## 5. 自动启动 Docker/vLLM 并寻优

先看计划和自检，然后在 tmux 中运行全部 phase：

```bash
python3 inferbench.py task start --name w4a8-search -- \
  search \
  --config configs/experiments/qwen38-w4a8-speed2k.json \
  --phase all
```

`all` 的流程是：

```text
冻结 fast workload
  → phase0 基线
  → phase1 候选测试与选优
  → phase2 候选测试与选优
  → 保存 best_candidate.json
```

每个 case 都单独启动服务、健康检查、执行 AISBench、提取指标、停止本 case 容器。清理逻辑只操作带本次 ownership label 的容器，不会按镜像名批量停止其他人的服务。

自动搜索具有防重复机制：同一配置、模型、数据、镜像和 phase 已完整完成时，再执行同一命令只返回已有目录，不重复跑。确实要重测时显式加 `--new-run`。

## 6. 最终 baseline/best 公平 A/B

搜索完成后，用同一个 `final` workload 分别测试原始 baseline 和 best：

```bash
python3 inferbench.py search \
  --config configs/experiments/qwen38-w4a8-speed2k.json \
  --phase final-baseline

python3 inferbench.py search \
  --config configs/experiments/qwen38-w4a8-speed2k.json \
  --phase final
```

- `final-baseline` 强制读取实验的原始 `baseline`。
- `final` 读取同一实验身份目录下的 `best_candidate.json`。
- 两边共用冻结后的 final workload。
- workload hash 一致时，第二条命令会自动生成 `comparison.csv/json/md`。

## 7. 构造 100K 输入数据

```bash
python3 inferbench.py data long-context \
  --config configs/experiments/qwen38-bf16-long100k.json \
  --source data/performance/throughput_2k_high_entropy.jsonl \
  --source data/performance/throughput_2k_low_entropy.jsonl
```

目标路径来自 `configs/workloads/long100k-c8.json`。生成：

- `data/performance/long100k_c8/requests.jsonl`
- 同目录 manifest，记录来源、token 长度、seed 和哈希。

已有目标默认不覆盖；确认要重新生成时加 `--force`。

## 8. tmux 后台任务

```bash
# 查看本项目后台任务
python3 inferbench.py task status

# 看某个任务最近 300 行屏幕输出
python3 inferbench.py task logs --name w4a8-search --lines 300

# 进入任务
python3 inferbench.py task attach --name w4a8-search

# 发送 Ctrl-C；自动搜索会清理自己创建的容器
python3 inferbench.py task stop --name w4a8-search
```

tmux session 自动加 `inferbench-` 前缀。已经在 tmux 内时，优先用 `task logs`，不要再次嵌套 attach。

## 9. 不重跑的报告操作

```bash
# 修改报告生成代码后重建某次报告
python3 inferbench.py report refresh --session <完整session目录>

# 修改 SLO 后，用已有原始数据重判
python3 inferbench.py report reclassify \
  --config configs/experiments/qwen38-bf16-long100k.json \
  --root <某次session或其上级目录>

# 手动指定 final-baseline 和 final 目录生成 A/B
python3 inferbench.py report compare \
  --baseline <final-baseline-session> \
  --candidate <final-session>
```

