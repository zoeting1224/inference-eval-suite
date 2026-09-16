# Local datasets

原始 benchmark 数据不提交到公开仓库。请将有权使用的数据放到配置指定位置：

```text
data/accuracy/gsm8k/test.jsonl
data/performance/throughput_2k_high_entropy.jsonl
data/performance/throughput_2k_low_entropy.jsonl
```

100K workload 通过 `inferbench.py data long-context` 从性能源数据生成，默认写入：

```text
data/performance/long100k_c8/requests.jsonl
```

服务器 `/mnt/data1/zt/benchmark/inference_eval_suite` 已保留当前本地数据；`.gitignore` 只阻止公开提交，不会删除文件。

