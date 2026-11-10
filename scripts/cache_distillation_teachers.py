"""Generate train-only soft probabilities from pinned image-only teachers."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for root in (PROJECT_ROOT, PROJECT_ROOT / 'src'):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from cxr_project.distillation_teachers import build_teacher_cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student-config', type=Path, default=PROJECT_ROOT / 'configs/distill_convnext_base_seed42.yaml')
    parser.add_argument('--teachers', type=Path, default=PROJECT_ROOT / 'configs/distillation_teachers_7779.json')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'data/distillation/teachers_7779_train.npz')
    parser.add_argument('--device', choices=('cuda', 'mps', 'cpu', 'auto'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    args = parser.parse_args(argv)
    result = build_teacher_cache(args.student_config, args.teachers, args.output, device_name=args.device,
                                 batch_size=args.batch_size, num_workers=args.num_workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
