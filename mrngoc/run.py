"""CLI chạy TF-Mamba gốc.

Ví dụ:
    python run.py --dataset hust  --raw-root /path/to/HUST-HAR
    python run.py --dataset all   --hust /p/HUST --uthar /p/UT --ntufi /p/NTU
Lưu ý: cần GPU + mamba-ssm (CUDA). Trên Windows không có CUDA-mamba thì dùng Kaggle.
"""
import argparse
import io
import json
import sys

# Windows console -> UTF-8 (tránh lỗi khi print)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch

from train import run_all, run_dataset, SEED, NORM_MODE, MERGE_VAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True,
                    choices=['hust', 'uthar', 'ntufi', 'all'])
    ap.add_argument('--raw-root', default=None,
                    help='root cua dataset (khi --dataset != all)')
    ap.add_argument('--hust', default=None)
    ap.add_argument('--uthar', default=None)
    ap.add_argument('--ntufi', default=None)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--norm-mode', default=NORM_MODE, choices=['author', 'double'],
                    help="author=SenseFi norm (dung tac gia) | double=+z-norm")
    ap.add_argument('--merge-val', action='store_true', default=MERGE_VAL,
                    help="CHI UT-HAR: gop X_val vao test (mac dinh khong, giong git SenseFi)")
    ap.add_argument('--out', default='results.json')
    args = ap.parse_args()

    if args.dataset == 'all':
        specs = {k: v for k, v in
                 (('hust', args.hust), ('uthar', args.uthar), ('ntufi', args.ntufi))
                 if v}
        if not specs:
            ap.error('--dataset all can it nhat mot trong --hust/--uthar/--ntufi')
        results = run_all(specs, device=args.device, seed=args.seed,
                          norm_mode=args.norm_mode, merge_val=args.merge_val)
    else:
        if not args.raw_root:
            ap.error('--raw-root bat buoc khi --dataset != all')
        results = [run_dataset(args.dataset, args.raw_root, device=args.device,
                               seed=args.seed, norm_mode=args.norm_mode,
                               merge_val=args.merge_val)]

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == '__main__':
    main()
