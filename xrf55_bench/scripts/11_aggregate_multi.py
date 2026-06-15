"""Aggregate S4.1 multi-dataset results into a comparison table.

Scans a results root for metrics.json (e.g. s41_<dataset>_<mode>_p02/metrics.json),
extracts headline (last_model) accuracy / macro-F1 mean±std, and prints + saves a
markdown table grouped by dataset, comparing proc vs raw.

Usage:
    python xrf55_bench/scripts/11_aggregate_multi.py --root outputs
    python xrf55_bench/scripts/11_aggregate_multi.py --root /kaggle/working/outputs --out summary.md
"""
import argparse
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

_ORDER = {'hust': 0, 'uthar': 1, 'ntufi': 2}


def _mode_of(m: dict) -> str:
    bd = str(m.get('bench_dir', '')).replace('\\', '/').lower()
    if '/proc' in bd or bd.endswith('proc'):
        return 'proc'
    if '/raw' in bd or bd.endswith('raw'):
        return 'raw'
    return m.get('config', {}).get('data_mode', '?')


def _row(m: dict) -> dict:
    s = m.get('summary', {})
    seeds = m.get('config', {}).get('seeds', [])
    n = len(seeds)
    if n > 1:
        acc = (s.get('test_accuracy_mean', 0) * 100, s.get('test_accuracy_std', 0) * 100)
        f1  = (s.get('test_f1_macro_mean', 0) * 100, s.get('test_f1_macro_std', 0) * 100)
    else:                                    # single seed — pull from per_seed
        ps = next(iter(m.get('per_seed', {}).values()), {})
        acc = (ps.get('test_accuracy', 0) * 100, 0.0)
        f1  = (ps.get('test_f1_macro', 0) * 100, 0.0)
    return {
        'dataset': m.get('dataset', '?'),
        'mode':    _mode_of(m),
        'n_seeds': n,
        'epochs':  m.get('config', {}).get('num_epochs', '?'),
        'acc':     acc,
        'f1':      f1,
        'params':  s.get('params_M', '?'),
    }


def main(root: Path, out: Path = None):
    metrics_files = sorted(root.rglob('metrics.json'))
    rows = []
    for mf in metrics_files:
        try:
            rows.append((_row(json.load(open(mf, encoding='utf-8'))), mf))
        except Exception as e:
            print(f'  skip {mf}: {e}')
    if not rows:
        print(f'No metrics.json found under {root}')
        return

    rows.sort(key=lambda rm: (_ORDER.get(rm[0]['dataset'], 9), rm[0]['mode']))

    lines = ['# S4.1 Multi-dataset Results (headline = last_model)', '',
             '| Dataset | Mode | Seeds | Ep | Accuracy (%) | Macro-F1 (%) | Params(M) |',
             '|---|---|---|---|---|---|---|']
    for r, _ in rows:
        acc = f"{r['acc'][0]:.2f} ± {r['acc'][1]:.2f}" if r['n_seeds'] > 1 else f"{r['acc'][0]:.2f}"
        f1  = f"{r['f1'][0]:.2f} ± {r['f1'][1]:.2f}"  if r['n_seeds'] > 1 else f"{r['f1'][0]:.2f}"
        lines.append(f"| {r['dataset']} | {r['mode']} | {r['n_seeds']} | {r['epochs']} | "
                     f"{acc} | {f1} | {r['params']} |")

    table = '\n'.join(lines)
    print(table)
    print(f'\n({len(rows)} run(s) aggregated from {root})')

    if out:
        out.write_text(table + '\n', encoding='utf-8')
        print(f'saved -> {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='outputs', help='results root to scan recursively')
    ap.add_argument('--out', default=None, help='optional markdown output path')
    a = ap.parse_args()
    main(Path(a.root), Path(a.out) if a.out else None)
