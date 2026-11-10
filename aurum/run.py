"""Command entry point for the executed AURUM research campaign."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    fit = sub.add_parser('anchor-ablation')
    fit.add_argument('--target', type=int, required=True)
    fit.add_argument('--peers', nargs=2, type=int, required=True)
    fit.add_argument('--device', default='cuda:0')
    fit.add_argument('--output', required=True)
    fit.add_argument('--ranks', type=int, nargs='+', default=[32, 128])
    fit.add_argument('--epochs', type=int, default=40)
    fit.add_argument('--patience', type=int, default=10)
    fit.add_argument('--lr', type=float, default=.001)
    fit.add_argument('--weight-decay', type=float, default=0.)
    fit.add_argument('--batch-size', type=int, default=32)
    fit.add_argument('--tv-epochs', type=int, default=15)
    fit.add_argument('--seed', type=int, default=20261110)
    fit.add_argument('--inner-fold', type=int, default=0)
    args = parser.parse_args()
    if args.action == 'anchor-ablation':
        from aurum.sprint import run_cell
        run_cell(args)


if __name__ == '__main__':
    main()
