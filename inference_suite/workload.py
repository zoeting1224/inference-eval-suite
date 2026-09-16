"""Freeze exact request payloads; render generic AISBench configs per dataset."""
import hashlib
import json
from pathlib import Path

from .config import digest, save


def resolve(root, path):
    p = Path(path).expanduser()
    return p if p.is_absolute() else Path(root) / p


def identity(root, c, require=False):
    """Fingerprint data AND small model/tokenizer metadata, not just display names."""
    files = {}
    paths = [resolve(root, d["path"]) for d in c["workload"]["datasets"]]
    for base in {c["model"]["path"], c["model"]["tokenizer"]}:
        for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "generation_config.json"):
            p = resolve(root, base) / name
            if p.is_file():
                paths.append(p)
    for p in paths:
        if p.is_file():
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            files[str(p)] = h.hexdigest()
        elif require:
            raise ValueError(f"missing input: {p}")
        else:
            files[str(p)] = "MISSING (plan only)"
    model_root = resolve(root, c["model"]["path"])
    weights = {}
    if model_root.is_dir():
        for pattern in ("*.safetensors", "*.bin", "*.safetensors.index.json"):
            for p in sorted(model_root.glob(pattern)):
                stat = p.stat()
                weights[p.name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {"config": digest(c), "files": files, "weight_file_metadata": weights}


def freeze(root, c, mode, directory):
    w = c["workload"]
    directory = Path(directory)
    datasets = []
    tokenizer = None
    if w.get("input_tokens") is not None:
        # Run with the AISBench Conda Python; no network download or remote-code trust by default.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(c["model"]["tokenizer"], local_files_only=True,
                         trust_remote_code=c["model"].get("trust_remote_code", False))
    for ds in w["datasets"]:
        rows = []
        source = resolve(root, ds["path"])
        with source.open() as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)
                text = raw[ds.get("text_field", "question")]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{source}: nonempty text required")
                item = {"question": text, "max_out_len": w["output_tokens"],
                        "request_id": f"{ds['name']}-{len(rows):06d}"}
                if tokenizer is not None:
                    if c["benchmark"]["model_class"] == "VLLMCustomAPIChat":
                        # Count token IDs, not BatchEncoding fields (Transformers 5 defaults).
                        ids = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                            tokenize=True, add_generation_prompt=True, return_dict=False,
                            **w.get("chat_template_kwargs", {}))
                    else:
                        ids = tokenizer.encode(text, add_special_tokens=True)
                    item["input_tokens"] = len(ids)
                    if abs(len(ids) - w["input_tokens"]) > w.get("input_tolerance", 0):
                        raise ValueError(f"{source} row {len(rows)}: {len(ids)} tokens, expected {w['input_tokens']}; no automatic truncation")
                rows.append(item)
                if len(rows) == w[mode]["prompts_per_dataset"]:
                    break
        if len(rows) != w[mode]["prompts_per_dataset"]:
            raise ValueError(f"{source}: needs {w[mode]['prompts_per_dataset']} rows, found {len(rows)}")
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / (ds["name"] + ".jsonl")
        target.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
        datasets.append({"name": ds["name"], "file": str(target.resolve()), "sha256": digest(rows),
                         "prompts": len(rows), "input_tokens": [r.get("input_tokens") for r in rows]})
    profile = {"mode": mode, "concurrency": w["concurrency"], "output_tokens": w["output_tokens"],
               "repeats": w[mode]["repeats"], "warmups": w[mode]["warmups"],
               "cache_policy": w["cache_policy"], "generation_kwargs": w["generation_kwargs"],
               "chat_template_kwargs": w.get("chat_template_kwargs", {}),
               "request_order": "first_n_per_dataset; datasets run sequentially",
               "traffic": "closed_loop, unlimited request_rate; one wave when prompts=concurrency",
               "datasets": datasets}
    # Absolute output paths vary by session and are not experiment conditions.
    canonical = dict(profile, datasets=[{k: v for k, v in d.items() if k != "file"} for d in datasets])
    profile["workload_hash"] = digest(canonical)
    save(directory / "workload_profile.json", profile)
    return profile


def render(c, profile, ds, directory):
    directory = Path(directory)
    (directory / "models").mkdir(parents=True, exist_ok=True)
    (directory / "datasets").mkdir(exist_ok=True)
    b, s, w = c["benchmark"], c["service"], c["workload"]
    kwargs = dict(w["generation_kwargs"])
    if w.get("chat_template_kwargs"):
        kwargs["chat_template_kwargs"] = w["chat_template_kwargs"]
    # AISBench loops over range(retry): 1 means one attempt, no extra retries;
    # 0 would silently send no requests at all (including warmup).
    options = dict(attr="service", abbr="target", path=c["model"]["tokenizer"],
        model=c["model"]["served_name"], stream=True, request_rate=0, use_timestamp=False,
        retry=1, api_key="", host_ip=s["host"], host_port=s["port"], url="",
        max_out_len=w["output_tokens"], batch_size=w["concurrency"],
        trust_remote_code=c["model"].get("trust_remote_code", False), generation_kwargs=kwargs)
    # Backend-version-specific settings are explicit config rather than guessed flags.
    for key, value in b.get("model_options", {}).items():
        if key in options or key == "type":
            raise ValueError(f"benchmark.model_options cannot override controlled field: {key}")
        options[key] = value
    text = f"from ais_bench.benchmark.models import {b['model_class']}\n"
    text += f"models = [{options!r}]\nmodels[0]['type'] = {b['model_class']}\n"
    (directory / "models" / "target.py").write_text(text)
    text = """from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import CustomDataset
"""
    text += f"""workload_datasets = [dict(
    abbr={ds['name']!r}, type=CustomDataset, path={ds['file']!r}, meta_path='',
    reader_cfg=dict(input_columns=['question', 'max_out_len'], output_column=None),
    infer_cfg=dict(prompt_template=dict(type=PromptTemplate, template='{{question}}'),
                   retriever=dict(type=ZeroRetriever), inferencer=dict(type=GenInferencer)),
    eval_cfg=dict())]
"""
    (directory / "datasets" / "workload.py").write_text(text)
    return directory
