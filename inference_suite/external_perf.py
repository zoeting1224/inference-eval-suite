#!/usr/bin/env python3
"""Benchmark an already-running vLLM endpoint with one fixed config.

Unlike autotune.py run, this never calls docker run/stop. The endpoint and
AISBench remain externally managed (for example by test(1).txt).
"""
import argparse, csv, datetime, json, sys
from pathlib import Path
from .config import load, validate, save
from .workload import freeze, render
from .metrics import extract
from .selection import evaluate
from .runner import run_benchmark, log
from .reporting import report

def main(argv=None, root=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--mode', choices=('fast','final'), default='final')
    ap.add_argument('--port', type=int, help='override configured endpoint port')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args(argv)
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    c = validate(load(args.config))
    if args.port is not None:
        c['service']['port'] = args.port
    out = args.output or (root / 'runs' / 'performance' / c['name'] / ('external-' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    profile = freeze(root, c, args.mode, out / 'workload')
    save(out / 'manifest.json', {'kind':'external-benchmark','mode':args.mode,'config':c,'workload_hash':profile['workload_hash']})
    rows=[]
    repeats=[]
    for repeat in range(profile['repeats']):
      for ds in profile['datasets']:
        label=f"repeat_{repeat+1}/{ds['name']}"
        work=out/label; cfg=render(c,profile,ds,work/'config')
        # AISBench model config is generated from c; replace only endpoint port.
        model_file=work/'config'/'models'/'target.py'
        text=model_file.read_text().replace("'host_port': 8004", "'host_port': %d" % c['service']['port'])
        model_file.write_text(text)
        argv=[c['benchmark']['binary'],'--config-dir',str(cfg),'--models','target','--datasets','workload','--mode','perf','--num-prompts',str(ds['prompts']),'--num-warmups',str(profile['warmups']),'--work-dir',str(work/'aisbench')]
        log(f'EXTERNAL endpoint {c["service"]["host"]}:{c["service"]["port"]}; {ds["name"]}: {" ".join(argv)}')
        rc=run_benchmark(argv,work/'aisbench.log',c['benchmark']['timeout_s'])
        if rc: raise RuntimeError(f'AISBench failed ({rc}); see {work/"aisbench.log"}')
        metrics, requests=extract(work/'aisbench',expected=ds['prompts'],output_tokens=profile['output_tokens'],input_tokens=c['workload'].get('input_tokens'),input_tolerance=c['workload'].get('input_tolerance',0),concurrency=profile['concurrency'],time_scale=c['benchmark'].get('detail_time_scale',1.0))
        save(work/'metrics.json',metrics)
        with (work/'requests.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(requests[0])); w.writeheader(); w.writerows(requests)
        repeats.append({'name':label,'metrics':metrics})
    row={'case':'external_000','case_hash':'external','phase':'external','mode':args.mode,
         'config':c['baseline'],'repeats':repeats,'evaluation':evaluate(repeats,c['selection'])}
    rows.append(row)
    save(out/'result.json',row); save(out/'slo_report.json',row['evaluation']); report(out,rows,c['selection'])
    print(out)
if __name__=='__main__':
    try: main()
    except Exception as e: print(f'ERROR: {type(e).__name__}: {e}',file=sys.stderr); sys.exit(1)
