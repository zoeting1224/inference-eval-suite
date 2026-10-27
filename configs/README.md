# 配置索引

日常使用只改 JSON，不改 `inference_suite/` 代码。

| 想修改的内容 | 文件位置 |
|---|---|
| 选用哪些模型/硬件/workload/search/SLO | `experiments/*.json` |
| 模型目录、容器内路径、served name、量化参数 | `models/*.json` |
| Docker 镜像、NPU 卡号、TP/DP、端口 | `hardware/*.json` |
| AISBench Conda Python、命令、超时 | `benchmarks/aisbench.json` |
| 数据路径、并发、输入/输出 token、warmup/repeat | `workloads/*.json` |
| 搜哪些 vLLM 参数及候选值 | `search/*.json` |
| SLO 达标条件和排名指标 | `selection/*.json` |
| 精度数据、API 地址、抽样数、生成参数 | `accuracy/*.json` |

`experiments/*.json` 使用 `extends` 从前到后合并配置，后出现的字段覆盖前面的字段；实验文件自己的字段最后覆盖。因此某个实验的特殊参数应直接写在实验文件的 `baseline.args` 中。

## 搜索预设

| 文件 | 用途 |
|---|---|
| `search/general.json` | 最小通用搜索，只调 batch token budget 与显存比例 |
| `search/qwen-mtp.json` | Qwen 长上下文的精简 MTP 搜索 |
| `search/qwen-comprehensive.json` | 旧 `search_space.env` 的完整迁移版：MTP、Batch 网格、KV/Cache、Graph，以及默认关闭的 HCCL/OMP/CPU binding |

Qwen3.8 W8A8 的开箱即用入口是：

```bash
python3 inferbench.py plan --config configs/experiments/qwen38-w8a8-comprehensive.json
python3 inferbench.py search --config configs/experiments/qwen38-w8a8-comprehensive.json --phase all
```

`plan` 会显示不符合当前 workload 的候选并说明跳过原因。`phase5` 默认关闭；高收益阶段完成后，可显式执行 `--phase phase5`。换成其他 Qwen 后端时，必须先核对 `args.speculative-config.method`。

