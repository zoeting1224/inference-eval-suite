#!/usr/bin/env python3
"""Construct long-input performance requests from existing JSONL texts.

Never edits source datasets; never downloads a tokenizer. Generated data is a
synthetic concatenation workload, NOT an accuracy benchmark or official split.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from .config import load, save, validate
from .workload import resolve


def fit(tokenizer, text, target, chat, template_kwargs, tolerance=0):
    ids = tokenizer.encode(text, add_special_tokens=False)
    def measure(value):
        # Count token IDs, not BatchEncoding fields (Transformers 5 defaults).
        return len(tokenizer.apply_chat_template([{"role": "user", "content": value}],
                 tokenize=True, add_generation_prompt=True, return_dict=False,
                 **template_kwargs)) if chat else len(tokenizer.encode(value, add_special_tokens=True))
    end = min(len(ids), target)
    visited = set()
    closest_under = None
    def remember(value, count):
        nonlocal closest_under
        if count < target and (closest_under is None or count > closest_under[1]):
            closest_under = (value, count)

    checked_suffixes = set()
    def fill_boundary(base=None):
        # Some decoded token prefixes skip the target on re-encoding (e.g.
        # 99999 -> 100002). A tiny deterministic suffix is valid for this
        # synthetic performance workload; always re-measure the whole template.
        base = closest_under if base is None else base
        if base is None or target - base[1] > 32:
            return None
        value, count = base
        suffixes = [".", " x", " 0", "\n0", "a", " ", "\n"]
        suffixes += [" x" * n for n in range(2, target - count + 5)]
        for suffix in suffixes:
            candidate = value + suffix
            if candidate in checked_suffixes:
                continue
            checked_suffixes.add(candidate)
            actual = measure(candidate)
            if abs(actual - target) <= tolerance:
                return candidate, actual
        return None

    for _ in range(20):
        if end in visited or end <= 0 or end > len(ids):
            break
        visited.add(end)
        result = tokenizer.decode(ids[:end], skip_special_tokens=False)
        count = measure(result)
        if abs(count-target) <= tolerance:
            return result, count
        remember(result, count)
        end += target-count
    filled = fill_boundary()
    if filled is not None:
        return filled
    # Decode/encode round trips can merge at a boundary. Search a bounded local window.
    sizes = range(max(1, end-32), min(len(ids), end+32)+1)
    for size in sorted(sizes, key=lambda size: (abs(size-end), size)):
        result = tokenizer.decode(ids[:size], skip_special_tokens=False)
        count = measure(result)
        if abs(count-target) <= tolerance:
            return result, count
        remember(result, count)
        # A slightly shorter prefix can have a cleaner byte/BPE boundary than
        # the closest one, allowing an exactly sized suffix.
        if 0 < target - count <= 4:
            filled = fill_boundary((result, count))
            if filled is not None:
                return filled
    filled = fill_boundary()
    if filled is not None:
        return filled
    raise ValueError("cannot reach exact template-inclusive token length; inspect tokenizer/template or explicitly set input_tolerance")


def main(argv=None, root=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--source", type=Path, action="append", required=True)
    p.add_argument("--text-field", default="question")
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--force", action="store_true", help="explicitly replace previously generated target, never source files")
    a = p.parse_args(argv)
    project_root = Path(root) if root else Path(__file__).resolve().parents[1]
    c = validate(load(a.config))
    python = Path(c["benchmark"]["python"]).resolve()
    if Path(sys.executable).resolve() != python:
        if not python.is_file():
            raise ValueError(f"configured Python not found: {python}")
        os.execv(str(python), [str(python), str(project_root / "inferbench.py"), *sys.argv[1:]])
    w = c["workload"]
    if w.get("input_tokens") is None or len(w["datasets"]) != 1:
        raise ValueError("preparation requires input_tokens and exactly one target dataset")
    root = project_root
    target = resolve(root, w["datasets"][0]["path"])
    if target.resolve() in [x.resolve() for x in a.source]:
        raise ValueError("target cannot be a source file")
    if target.exists() and not a.force:
        raise ValueError(f"target already exists: {target}; use another path or explicit --force")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(c["model"]["tokenizer"], local_files_only=True,
                      trust_remote_code=c["model"].get("trust_remote_code", False))
    pool, hashes = [], {}
    for path in a.source:
        hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        for index, line in enumerate(path.read_text().splitlines()):
            raw = json.loads(line)
            text = raw[a.text_field]
            if isinstance(text, str) and text.strip():
                pool.append((str(path.resolve()), index, text))
    count = max(w["fast"]["prompts_per_dataset"], w["final"]["prompts_per_dataset"])
    if len(pool) < count:
        raise ValueError("need at least one distinct source prefix per generated request")
    rng = random.Random(a.seed)
    rng.shuffle(pool)
    lengths = [len(tokenizer.encode(item[2], add_special_tokens=False)) for item in pool]
    rows, provenance = [], []
    for i in range(count):
        # Different initial source for every request; rest is shuffled, reuse is documented.
        order = [i] + rng.sample([j for j in range(len(pool)) if j != i], len(pool)-1)
        parts, refs, total, cursor = [], [], 0, 0
        while total < w["input_tokens"] + 1024:
            j = order[cursor % len(order)]
            source, index, text = pool[j]
            parts.append(text)
            refs.append({"source": source, "row": index})
            total += max(1, lengths[j])
            cursor += 1
        text, actual = fit(tokenizer, "\n\n".join(parts), w["input_tokens"],
                          c["benchmark"]["model_class"] == "VLLMCustomAPIChat",
                          w.get("chat_template_kwargs", {}), w.get("input_tolerance", 0))
        rows.append({w["datasets"][0].get("text_field", "question"): text, "input_tokens": actual,
                     "id": f"stitched-{i}", "max_out_len": w["output_tokens"]})
        provenance.append({"id": f"stitched-{i}", "sources_before_final_truncation": refs,
                           "corpus_reused_within_request": cursor > len(pool), "actual_input_tokens": actual})
        print(f"Prepared {i+1}/{count}: {actual} tokens", flush=True)
    if len({hashlib.sha256(r[w['datasets'][0].get('text_field', 'question')].encode()).hexdigest() for r in rows}) != count:
        raise ValueError("constructed requests are not distinct")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows))
    tmp.replace(target)
    save(target.with_suffix(".manifest.json"), {"kind": "synthetic concatenation, performance only",
        "length_adjustment": "token-prefix truncation; bounded deterministic suffix if needed; full-template token count verified",
        "seed": a.seed, "tokenizer": c["model"]["tokenizer"], "source_hashes": hashes,
        "target_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "requests": provenance})
    print(target)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
