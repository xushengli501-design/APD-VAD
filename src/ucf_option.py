import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='VadCLIP')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=2, type=int)
parser.add_argument('--attn-window', default=8, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=14, type=int)

parser.add_argument('--max-epoch', default=15, type=int)
parser.add_argument('--model-path', default='model/ucf_best_map.pth')
parser.add_argument('--model-bundle-path', default='', type=str)
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='model/ckpt_ucf.pth')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--train-list', default='list/ucf_CLIP_rgb.csv')
parser.add_argument('--test-list', default='list/ucf_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='list/gt_ucf.npy')
parser.add_argument('--gt-segment-path', default='list/gt_segment_ucf.npy')
parser.add_argument('--gt-label-path', default='list/gt_label_ucf.npy')

parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--scheduler-rate', default=0.1, type=float)
parser.add_argument('--scheduler-milestones', default=[4, 8], nargs='+', type=int)
parser.add_argument('--text-feature-version', default='v1', choices=['v1', 'v2', 'ucf_text', 'ucf_text_v2'],
                    help='Select UCF text feature set. v1=ucf_text, v2=ucf_text_v2')
parser.add_argument('--text-feature-dir', default=None, type=str)
parser.add_argument('--snippet-text-feature-dir', default=None, type=str)
parser.add_argument('--use-snippet-text-gating', action='store_true')
parser.add_argument('--use-mil-text-gating', action='store_true')
parser.add_argument('--binary-priority', action='store_true')
parser.add_argument('--mil-text-temp', default=0.1, type=float)
parser.add_argument('--use-adaptive-mil', action='store_true')
parser.add_argument('--adaptive-mil-temp', default=0.1, type=float)
parser.add_argument('--use-motion-refine', action='store_true')
parser.add_argument('--use-category-refine', action='store_true')
parser.add_argument('--category-loss-weight', default=0.1, type=float)
parser.add_argument('--prototype-temp', default=0.07, type=float)
parser.add_argument('--use-debiased-causal-graph', default=True, action='store_true')
parser.add_argument('--debiased-graph-threshold', default=0.2, type=float)
parser.add_argument('--gaussian-sigma', default=0.0, type=float)
parser.add_argument('--hard-neg-weight', default=0.0, type=float)
parser.add_argument('--hard-neg-topk', default=3, type=int)
parser.add_argument('--hard-neg-margin', default=0.5, type=float)
parser.add_argument('--dcsa-weight', default=0.0, type=float)
parser.add_argument('--dcsa-temperature', default=0.07, type=float)

parser.add_argument('--use-sgnm', action='store_true')
parser.add_argument('--sgnm-compact-weight', default=0.1, type=float)
parser.add_argument('--sgnm-consist-weight', default=0.1, type=float)
parser.add_argument('--sgnm-num-queries', default=16, type=int)
parser.add_argument('--sgnm-decoder-depth', default=4, type=int)
parser.add_argument('--sgnm-normal-ratio', default=0.8, type=float)

parser.add_argument('--text-sep-weight', default=0.0, type=float,
                    help='Weight for text separation loss (pushes Normal text away from anomaly classes).')
parser.add_argument('--teacher-epochs', default=1, type=int)
parser.add_argument('--teacher-lr', default=2e-5, type=float)
parser.add_argument('--teacher-model-path', default='model/ucf_dsanet_v3_teacher.pth', type=str)
parser.add_argument('--train-teacher', action='store_true', default=True)
parser.add_argument('--distill-weight', default=0.5, type=float)
parser.add_argument('--loss1-weight', default=0.8, type=float)
parser.add_argument('--loss2-weight', default=1.1, type=float)
parser.add_argument('--temporal-consistency-weight', default=0.2, type=float)
parser.add_argument('--event-kd-weight', default=0.3, type=float)
parser.add_argument('--event-trend-weight', default=0.15, type=float)

parser.add_argument('--text-adapt-until', default=3, type=int)
parser.add_argument('--t-w', default=0.1, type=float)

parser.add_argument('--DNP-use', default=True, type=lambda x: x.lower() != 'false')
parser.add_argument('--num-prototypes', default=16, type=int)
parser.add_argument('--decoder-depth', default=8, type=int)
parser.add_argument('--normal-selection-ratio', default=0.8, type=float)

parser.add_argument('--temp', default=5.0, type=float)
parser.add_argument('--use-train-visual-retrieval', action='store_true')
parser.add_argument('--retrieval-target', default='visual', choices=['visual'])
parser.add_argument('--retrieval-fuse', default='add', choices=['add'])
parser.add_argument('--retrieval-topk', default=5, type=int)
parser.add_argument('--retrieval-temp', default=0.07, type=float)
parser.add_argument('--retrieval-weight', default=0.3, type=float)
parser.add_argument('--pretrained-path', default='model/model_ucf.pth', type=str)
parser.add_argument('--log-path', default='model/ucf_train.log', type=str)


_UCF_TEXT_DIR_MAP = {
    'v1': (
        'data/ucf_text',
        'data/ucf_text_llm_snippets',
    ),
    'ucf_text': (
        'data/ucf_text',
        'data/ucf_text_llm_snippets',
    ),
    'v2': (
        'data/ucf_text_v2',
        'data/ucf_text_v2_snippets',
    ),
    'ucf_text_v2': (
        'data/ucf_text_v2',
        'data/ucf_text_v2_snippets',
    ),
}


def resolve_text_feature_dirs(args):
    version = getattr(args, 'text_feature_version', 'v1')
    default_text_dir, default_snippet_dir = _UCF_TEXT_DIR_MAP[version]
    if not getattr(args, 'text_feature_dir', None):
        args.text_feature_dir = default_text_dir
    if not getattr(args, 'snippet_text_feature_dir', None):
        args.snippet_text_feature_dir = default_snippet_dir
    args.text_feature_dir = str(Path(args.text_feature_dir))
    args.snippet_text_feature_dir = str(Path(args.snippet_text_feature_dir))
    return args
