# 结果文件说明

所有新结果都在 `runs/`，代码不会再写旧项目的 `results/`、`state/` 或 `outputs/`。

## 固定精度测试

```text
runs/accuracy/<run>/
├── manifest.json       # 数据哈希、抽样序号、生成参数、endpoint
├── predictions.jsonl   # 每题答案、gold、正确性、耗时、错误；也是续跑状态
├── summary.json        # 程序读取的精度汇总
└── summary.md          # 人工直接查看
```

先看 `summary.md`；需要逐题排查时看 `predictions.jsonl`。BF16/W4A8 对比结果在指定的 comparison 目录。

## 固定性能测试

```text
runs/performance/<run>/
├── manifest.json
├── workload/workload_profile.json
├── repeat_1/<dataset>/
│   ├── aisbench.log
│   ├── metrics.json
│   └── requests.csv
├── result.json
├── slo_report.json
├── summary.csv
├── ranking.csv
├── summary.json
└── summary.md
```

- `requests.csv`：确认每条请求是否成功，并查看 input/output tokens、TTFT、TPOT、decode TPS、E2E。
- `metrics.json`：该 dataset/repeat 的汇总指标。
- `slo_report.json`：每项门槛的 PASS/FAIL。
- `summary.csv`：按执行顺序。
- `ranking.csv`：按目标函数排名；它不是另一组测量。
- `summary.md`：人读表格。

## 自动寻优

```text
runs/performance/<experiment>/<identity>/
├── latest -> runs/<最新session>
├── state/
│   ├── best_candidate.json
│   ├── final-baseline.json
│   └── final.json
└── runs/<session>/
    ├── master.log
    ├── session_status.json
    ├── workload/workload_profile.json
    ├── phase*_plan.json
    ├── runs/<case>/
    │   ├── launch.sh
    │   ├── vllm.log
    │   ├── cleanup.json
    │   ├── result.json
    │   └── repeat_N/<dataset>/...
    ├── summary.csv
    ├── ranking.csv
    └── summary.md
```

排错顺序：

1. `master.log` 看流程停在哪个 case；
2. case 下 `vllm.log` 看模型启动或运行错误；
3. `aisbench.log` 看客户端错误/超时；
4. `requests.csv` 看成功条数和逐路时延；
5. `cleanup.json` 确认模型是否由控制器正常清理。

`workload_profile.json` 会记录 mode、prompts、repeat、warmup、concurrency 和 workload hash，用于证明不同 case 使用相同测试条件。

