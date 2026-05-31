"""Driver for the rebuttal partial-correlation sweep.

Encodes (analysis, predictor, control, label) tuples and dispatches one
(sub, tuple) pair per SLURM task index.

Two analyses:

(A) Trained vs untrained — partial r (trained_pred, bio | untrained_pred):
    unique variance attributable to training-induced features.

(B) Layer-wise — for each DNN layer L ∈ {V1,V2,V4,IT,decoder}:
    - partial r (fusion@L, bio | DNN-only@L)  → LLM unique at layer L
    - partial r (fusion@L, bio | TEL)         → DNN-layer-L unique given LLM

Total tasks = 4*10 + 10*10 = 140  → SLURM array 0-139.
"""

import argparse
import os.path as op
import subprocess
import sys

# --- path configuration (portable defaults for public release) ---
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_DATA_ROOT = _os.environ.get('EEG_FUSION_DATA', _os.path.join(_REPO, 'data'))
# --- end path configuration ---


LAYERS = ['V1', 'V2', 'V4', 'IT', 'decoder']

TRAINED_VS_UNTRAINED = [
    # (predictor, control, label)
    ('cornet_s_r2_v3_e5mistral',
     'cornet_s_r2_v3_e5mistral_untrained',
     'cornet_s_untrained'),
    ('cornet_s_with_text_embedding_large_r2_v3_pc500',
     'cornet_s_untrained_with_text_embedding_large_r2_v3',
     'cornet_untrained_plus_TEL'),
    ('cornet_s_with_e5-mistral-7b-instruct_cleaned_r2_v3_e5mistral',
     'cornet_s_with_e5-mistral-7b-instruct_untrained_cleaned_r2_v3_e5mistral_untrained',
     'cornet_untrained_plus_e5mistral'),
    ('e5-mistral-7b-instruct_cleaned_r2_v3_e5mistral',
     'e5-mistral-7b-instruct_untrained_cleaned_r2_v3_e5mistral_untrained',
     'e5mistral_untrained'),
]

LAYERWISE = []
for L in LAYERS:
    fusion = f'cornet_s_with_text_embedding_large_r2_v3_layerwise_{L}'
    vision = f'cornet_s_r2_v3_layerwise_{L}'
    LAYERWISE.append((fusion, vision, f'vision_layer_{L}'))           # LLM unique at L
    LAYERWISE.append((fusion, 'text_embedding_large_r2_v3',
                      f'TEL_at_layer_{L}'))                            # DNN-layer unique given LLM

ALL_TUPLES = [('trained_vs_untrained', *t) for t in TRAINED_VS_UNTRAINED] + \
             [('layerwise', *t) for t in LAYERWISE]

SUBJECTS = list(range(1, 11))


def decode(task_id):
    """task_id ∈ [0, 139]: outer = tuple_idx, inner = subject."""
    n_tuples = len(ALL_TUPLES)
    n_subs = len(SUBJECTS)
    assert task_id < n_tuples * n_subs, (task_id, n_tuples, n_subs)
    tup_i = task_id // n_subs
    sub_i = task_id % n_subs
    analysis, predictor, control, label = ALL_TUPLES[tup_i]
    return analysis, predictor, control, label, SUBJECTS[sub_i]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task_id', type=int, default=-1)
    p.add_argument('--project_dir', type=str,
                   default=_DATA_ROOT)
    p.add_argument('--python', type=str,
                   default='/trinity/shared/easybuild/software/Anaconda3/2024.06-1/bin/python3')
    p.add_argument('--list', action='store_true', help='Just list tuples and exit')
    args = p.parse_args()

    if args.list:
        for i, t in enumerate(ALL_TUPLES):
            print(i, t)
        print(f'Total tuples: {len(ALL_TUPLES)}, '
              f'tasks (×{len(SUBJECTS)} subs): {len(ALL_TUPLES) * len(SUBJECTS)}')
        return

    analysis, predictor, control, label, sub = decode(args.task_id)
    print(f'task={args.task_id} analysis={analysis} sub={sub}')
    print(f'  predictor={predictor}')
    print(f'  control={control}  label={label}')

    here = op.dirname(op.abspath(__file__))
    cmd = [
        args.python, op.join(here, 'partial_correlation.py'),
        '--sub', str(sub),
        '--project_dir', args.project_dir,
        '--predictor', predictor,
        '--control', control,
        '--label', label,
    ]
    print('cmd:', ' '.join(cmd), flush=True)
    sys.exit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
