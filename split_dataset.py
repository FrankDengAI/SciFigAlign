import argparse
import random
from collections import defaultdict

from src.utils.io import ensure_dir, read_jsonl, write_jsonl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_jsonl', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--train_ratio', type=float, default=0.8)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--group_by_paper', action='store_true', default=True,
                   help='Paper-level split (default; matches paper protocol)')
    p.add_argument('--no_group_by_paper', action='store_false', dest='group_by_paper',
                   help='Figure-level random split (not used in paper)')
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    rows = read_jsonl(args.input_jsonl)
    rng = random.Random(args.seed)

    if args.group_by_paper:
        groups = defaultdict(list)
        for row in rows:
            groups[row['paper_id']].append(row)
        paper_ids = list(groups.keys())
        rng.shuffle(paper_ids)
        n = len(paper_ids)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)
        train_ids = set(paper_ids[:n_train])
        val_ids = set(paper_ids[n_train:n_train + n_val])
        test_ids = set(paper_ids[n_train + n_val:])
        train = [x for pid in train_ids for x in groups[pid]]
        val = [x for pid in val_ids for x in groups[pid]]
        test = [x for pid in test_ids for x in groups[pid]]
    else:
        rng.shuffle(rows)
        n = len(rows)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)
        train = rows[:n_train]
        val = rows[n_train:n_train + n_val]
        test = rows[n_train + n_val:]

    write_jsonl(f'{args.output_dir}/train.jsonl', train)
    write_jsonl(f'{args.output_dir}/val.jsonl', val)
    write_jsonl(f'{args.output_dir}/test.jsonl', test)
    print(f'Train={len(train)} Val={len(val)} Test={len(test)}')


if __name__ == '__main__':
    main()
