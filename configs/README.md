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

