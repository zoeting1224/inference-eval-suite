"""JSON composition, validation and deterministic staged search (stdlib only)."""
import copy
import hashlib
import itertools
import json
import math
import re
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load(path, stack=()):
    path = Path(path).resolve()
    if path in stack:
        raise ValueError(f"configuration inheritance cycle: {path}")
    data = json.loads(path.read_text())
    parents = data.pop("extends", [])
    if not isinstance(parents, list):
        raise ValueError("extends must be a list of relative JSON paths")
    result = {}
    for parent in parents:
        result = merge(result, load(path.parent / parent, (*stack, path)))
    return merge(result, data)


def positive(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def validate(c):
    required = {"name", "model", "service", "benchmark", "workload", "baseline", "search", "selection"}
    if set(c) != required:
        raise ValueError(f"top-level keys: missing={required-set(c)}, unknown={set(c)-required}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", c["name"]):
        raise ValueError("name must be a safe identifier")
    allowed = {
        "model": {"path", "container_path", "served_name", "tokenizer", "trust_remote_code", "revision", "docker_args"},
        "service": {"kind", "image", "command", "docker_args", "resource_ids", "host", "port", "health_timeout_s", "stop_timeout_s", "env"},
        "benchmark": {"python", "binary", "model_class", "timeout_s", "detail_time_scale", "model_options"},
        "workload": {"concurrency", "input_tokens", "input_tolerance", "output_tokens", "cache_policy", "generation_kwargs", "chat_template_kwargs", "datasets", "fast", "final"},
        "baseline": {"args", "env"}, "search": {"max_cases_per_phase", "phases"},
        "selection": {"constraints", "objectives"}}
    for section, keys in allowed.items():
        unknown = set(c[section])-keys
        if unknown:
            raise ValueError(f"unknown {section} fields: {unknown}")
    for key in ("path", "container_path", "served_name", "tokenizer"):
        if not c["model"].get(key):
            raise ValueError(f"model.{key} is required")
    if not isinstance(c["model"].get("docker_args", []), list):
        raise ValueError("model.docker_args must be an argv list")
    s = c["service"]
    if s.get("kind") != "docker":
        raise ValueError("only managed docker services are implemented; not arbitrary external backends")
    for key in ("image", "command", "docker_args", "resource_ids", "host", "port", "health_timeout_s"):
        if key not in s:
            raise ValueError(f"service.{key} is required")
    if not s["resource_ids"] or not all(re.fullmatch(r"[A-Za-z0-9_.-]+", x) for x in s["resource_ids"]):
        raise ValueError("service.resource_ids must identify the physical accelerators for locking")
    positive(s["port"], "service.port")
    positive(s["health_timeout_s"], "service.health_timeout_s")
    positive(s.get("stop_timeout_s", 45), "service.stop_timeout_s")
    if s["port"] > 65535 or s["host"] not in ("127.0.0.1", "localhost"):
        raise ValueError("managed Docker adapter currently requires a loopback host and valid port")
    if not isinstance(s["command"], list) or not s["command"]:
        raise ValueError("service.command must be an argv list")
    # Lifecycle flags are owned by the controller, not arbitrary Docker options.
    forbidden = {"--name", "--rm", "--detach", "-d", "--label", "-l", "--restart"}
    if any(x.split("=", 1)[0] in forbidden for x in s["docker_args"]):
        raise ValueError("docker_args cannot override container ownership/lifecycle")
    b, w = c["benchmark"], c["workload"]
    for key in ("python", "binary", "timeout_s", "model_class"):
        if key not in b:
            raise ValueError(f"benchmark.{key} is required")
    if b["model_class"] not in ("VLLMCustomAPIChat", "VLLMCustomAPI"):
        raise ValueError("implemented AISBench adapters: VLLMCustomAPIChat / VLLMCustomAPI")
    positive(w["concurrency"], "workload.concurrency")
    positive(w["output_tokens"], "workload.output_tokens")
    positive(b["timeout_s"], "benchmark.timeout_s")
    if w["output_tokens"] < 2:
        raise ValueError("output_tokens must be >=2 for decode measurement")
    if type(w.get("input_tolerance", 0)) is not int or w.get("input_tolerance", 0) < 0:
        raise ValueError("input_tolerance must be a nonnegative integer")
    if w["generation_kwargs"].get("ignore_eos") is not True:
        raise ValueError("fixed-length performance workloads require generation_kwargs.ignore_eos=true")
    if set(w["generation_kwargs"]) & {"max_tokens", "max_completion_tokens", "n", "stream", "model", "messages", "prompt"}:
        raise ValueError("generation_kwargs cannot override controlled request fields")
    if w["cache_policy"] not in ("cold", "warm"):
        raise ValueError("cache_policy must be cold or warm")
    if w.get("input_tokens") is not None:
        positive(w["input_tokens"], "workload.input_tokens")
    if not w["datasets"]:
        raise ValueError("workload.datasets is empty")
    names = [d["name"] for d in w["datasets"]]
    if len(set(names)) != len(names) or not all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", x) for x in names):
        raise ValueError("dataset names must be unique Python identifiers")
    for mode in ("fast", "final"):
        m = w[mode]
        if set(m) != {"prompts_per_dataset", "repeats", "warmups"}:
            raise ValueError(f"unknown/missing workload.{mode} fields")
        for key in ("prompts_per_dataset", "repeats"):
            positive(m[key], f"workload.{mode}.{key}")
        if m["prompts_per_dataset"] < w["concurrency"]:
            raise ValueError(f"{mode}: EACH sequential dataset needs at least concurrency requests")
        if type(m["warmups"]) is not int or m["warmups"] < 0:
            raise ValueError("warmups must be nonnegative")
    for group in ("args", "env"):
        if not isinstance(c["baseline"].get(group), dict):
            raise ValueError(f"baseline.{group} must be an object")
    phases = c["search"]["phases"]
    ids = [p["name"] for p in phases]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"phase[1-9][0-9]*", x) for x in ids):
        raise ValueError("search phase names must be unique phase1, phase2, ...")
    positive(c["search"]["max_cases_per_phase"], "search.max_cases_per_phase")
    for p in phases:
        if p["strategy"] not in ("one_at_a_time", "grid") or not p["parameters"]:
            raise ValueError("phase needs a strategy and nonempty parameters")
        for key, values in p["parameters"].items():
            if not re.fullmatch(r"(args|env)\.[A-Za-z0-9_-]+", key) or not isinstance(values, list) or not values:
                raise ValueError(f"invalid search dimension {key}")
        candidates(c["baseline"], p, c["search"]["max_cases_per_phase"])
    for key, limits in c["selection"].get("constraints", {}).items():
        if not limits or set(limits)-{"min", "max"}:
            raise ValueError(f"invalid constraint {key}")
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in limits.values()):
            raise ValueError("constraint values must be finite numbers")
    if not c["selection"]["objectives"]:
        raise ValueError("selection.objectives is empty")
    for obj in c["selection"]["objectives"]:
        if obj["direction"] not in ("min", "max"):
            raise ValueError("objective direction must be min/max")
    validate_case(c, c["baseline"])
    return c


def validate_case(c, candidate):
    args = candidate["args"]
    shutdown = args.get("shutdown-timeout")
    if shutdown is not None:
        if type(shutdown) is not int or shutdown < 0:
            raise ValueError("shutdown-timeout must be a nonnegative integer")
        if c["service"].get("stop_timeout_s", 45) < shutdown + 5:
            raise ValueError("service.stop_timeout_s must allow shutdown-timeout plus at least 5 seconds")
    for key in ("max-num-seqs", "max-model-len", "max-num-batched-tokens"):
        if key in args:
            positive(args[key], key)
    # Concurrency is an invariant of the workload, not a search dimension.
    if args.get("max-num-seqs", c["workload"]["concurrency"]) < c["workload"]["concurrency"]:
        raise ValueError("max-num-seqs below requested concurrency")
    length = c["workload"].get("input_tokens")
    if length and args.get("max-model-len", 0) < length + c["workload"]["output_tokens"]:
        raise ValueError("max-model-len must cover template-inclusive input + output")
    if c["workload"]["cache_policy"] == "cold":
        if args.get("enable-prefix-caching") is not False:
            raise ValueError("cold workload requires explicit enable-prefix-caching=false")
    if args.get("gpu-memory-utilization") is not None and not 0 < args["gpu-memory-utilization"] <= 1:
        raise ValueError("invalid gpu-memory-utilization")
    for key in ("host", "port", "served-model-name"):
        if key in args:
            raise ValueError(f"{key} is owned by service/model configuration")


def candidates(incumbent, phase, limit):
    dims = list(phase["parameters"].items())
    theoretical = (math.prod(len(v) for _, v in dims) if phase["strategy"] == "grid"
                   else sum(len(v) for _, v in dims))
    if theoretical + 1 > limit:
        raise ValueError(f"{phase['name']}: {theoretical+1} cases exceed cap {limit}")
    patches = ([dict(zip([k for k, _ in dims], values))
                for values in itertools.product(*(v for _, v in dims))]
               if phase["strategy"] == "grid" else [{k: v} for k, values in dims for v in values])
    result = [copy.deepcopy(incumbent)]  # Always remeasure the incumbent under this same workload.
    seen = {digest(incumbent)}
    for patch in patches:
        item = copy.deepcopy(incumbent)
        for key, value in patch.items():
            group, flag = key.split(".", 1)
            item[group][flag] = value
        if digest(item) not in seen:
            seen.add(digest(item))
            result.append(item)
    return result
