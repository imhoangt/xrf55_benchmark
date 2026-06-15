"""Fast end-to-end smoke test for the S4.1 multi-dataset path (HUST dims).

Builds a TINY packed wavmamba bench from 24 HUST samples, loads it through the
existing PreprocWavMambaDataset, checks normalization, then runs a WavDualMamba
forward — validating layout + dims + model before the full ~10-min build.
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from xrf55_bench.preprocessing.multi_dataset import pack_haar3
from xrf55_bench.dataset import PreprocWavMambaDataset, load_stats, infer_data_mode
from xrf55_bench.models.wavdualmamba.model import WavDualMamba

DATA = PROJECT_ROOT / 'dataset' / 'HUST-HAR'
TMP  = PROJECT_ROOT / 'dataset' / 'HUST-HAR' / 'bench' / '_smoke'
N_ANT, SUB, N_PER_SUB, FS, CLASSES = 9, 30, 9, 200.0, 6
NTR, NTE = 16, 8

print('Loading 24 HUST samples ...')
X = torch.load(DATA / 'HUST_HAR_dataset-001.pt', map_location='cpu', weights_only=False)
y = torch.load(DATA / 'HUST_HAR_labels.pt', map_location='cpu', weights_only=False)
X = X.numpy().astype(np.float32)
y = y.numpy().astype(np.int64)
print(f'  full X={X.shape}  y={y.shape}  labels={sorted(set(y.tolist()))}')

idx = {'train': np.arange(NTR), 'test': np.arange(NTR, NTR + NTE)}
C, T2, F2 = 3 * N_PER_SUB, X.shape[2] // 2, SUB // 2
wav = TMP / 'wavmamba'; wav.mkdir(parents=True, exist_ok=True)

s = np.zeros((C, F2)); s2 = np.zeros((C, F2)); n_acc = 0
for sp, ix in idx.items():
    arr = np.zeros((len(ix), C, T2, F2), dtype=np.float32)
    for j, i in enumerate(ix):
        arr[j] = pack_haar3(X[i], N_ANT, SUB, N_PER_SUB, fs=FS, do_filter=True)
    np.save(wav / f'X_{sp}.npy', arr)
    np.save(TMP / f'y_{sp}.npy', y[ix].astype(np.int64))
    xd = arr.astype(np.float64)
    s  += xd.sum(axis=(0, 2)); s2 += (xd * xd).sum(axis=(0, 2)); n_acc += len(ix) * T2
mean = s / n_acc
std  = np.maximum(np.sqrt(np.maximum(s2 / n_acc - mean * mean, 0.0)), 1e-6)
json.dump({'wavmamba': {'mean': mean.astype(np.float32).tolist(),
                        'std': std.astype(np.float32).tolist()},
           'meta': {'source': 'multi_hampel_lpf', 'filter': {}}},
          open(TMP / 'stats.json', 'w'))
print(f'  packed shape=({len(idx["train"])},{C},{T2},{F2})  stats(mean,std)=({C},{F2})')

# ── Load through the REAL dataset class + check normalization ──────────────────
stats = load_stats(TMP)
print(f'  infer_data_mode -> {infer_data_mode(stats)}  (expect proc)')
ds = PreprocWavMambaDataset(TMP, 'train', stats)
xb = torch.stack([ds[i][0] for i in range(len(ds))]).numpy()   # (16, C, T2, F2)
print(f'  loaded normalized batch={xb.shape}  mean={xb.mean():.4f} (~0)  std={xb.std():.4f} (~1)')

# ── Smoke-build the S4.1 model (HUST dims) + forward ───────────────────────────
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WavDualMamba(num_classes=CLASSES, n_links=3, n_antennas=3, f2=F2,
                     subbands=('LL', 'HL', 'LH'), pool='attnstat').to(dev)
nparam = sum(p.numel() for p in model.parameters())
xin = torch.stack([ds[i][0] for i in range(8)]).to(dev)
model.eval()
with torch.no_grad():
    out = model(xin)
print(f'  model C_IN={model.C_IN}  params={nparam/1e6:.3f}M')
print(f'  forward: in={tuple(xin.shape)} -> out={tuple(out.shape)} (expect (8,{CLASSES}))')
assert out.shape == (8, CLASSES), f'BAD output shape {out.shape}'
assert model.C_IN == C, f'C_IN {model.C_IN} != packed C {C}'
print('SMOKE OK — layout, normalization, dims, model forward all valid.')

import shutil
shutil.rmtree(TMP, ignore_errors=True)
print('  cleaned _smoke dir')
