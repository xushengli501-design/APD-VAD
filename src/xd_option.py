import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='VadCLIP')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=1, type=int)
parser.add_argument('--attn-window', default=64, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=7, type=int)

parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default='model/model_xd.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='model/checkpoint.pth')
parser.add_argument('--batch-size', default=96, type=int)
parser.add_argument('--train-list', default='list/xd_CLIP_rgb.csv')
parser.add_argument('--test-list', default='list/xd_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='list/gt.npy')
parser.add_argument('--gt-segment-path', default='list/gt_segment.npy')
parser.add_argument('--gt-label-path', default='list/gt_label.npy')

parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--scheduler-rate', default=0.1, type=float)
parser.add_argument('--scheduler-milestones', default=[3, 6, 10], nargs='+', type=int,
                    help='MultiStepLR milestones, e.g. --scheduler-milestones 3 6 10')
parser.add_argument('--use-motion-refine', action='store_true')
parser.add_argument('--hard-neg-weight', default=0.0, type=float)
parser.add_argument('--hard-neg-topk', default=3, type=int)
parser.add_argument('--hard-neg-margin', default=0.15, type=float)
parser.add_argument('--use-audio-aux', action='store_true')
parser.add_argument('--audio-root', default='data/xd_audio_vggish', type=str)
parser.add_argument('--audio-dim', default=128, type=int)
parser.add_argument('--audio-fusion-mode', default='normal', choices=['normal', 'identity', 'stats', 'cross_attn'], type=str)
parser.add_argument('--audio-cross-attn-heads', default=4, type=int)
parser.add_argument('--init-model-path', default='', type=str)
parser.add_argument('--freeze-clip', action='store_true')
parser.add_argument('--temporal-consistency-weight', default=0.0, type=float)
parser.add_argument('--caption-align-weight', default=0.0, type=float)
parser.add_argument('--snippet-gate-temperature', default=0.1, type=float)
parser.add_argument('--snippet-gate-residual', default=1.0, type=float)
parser.add_argument('--text-feature-version', default='v1', choices=['v1', 'xd_text', 'v2', 'xd_text_v2'])
parser.add_argument('--text-feature-dir', default=None, type=str)
parser.add_argument('--snippet-text-feature-dir', default=None, type=str)
parser.add_argument('--use-snippet-text-gating', action='store_true')
parser.add_argument('--use-debiased-causal-graph', action='store_true')
parser.add_argument('--debiased-graph-threshold', default=0.7, type=float)
parser.add_argument('--causal-repr-alpha', default=0.2, type=float)
parser.add_argument('--causal-repr-detach', action='store_true')
parser.add_argument('--use-rag', action='store_true')
parser.add_argument('--rag-topk', default=5, type=int)
parser.add_argument('--rag-max-bank-size', default=6000, type=int)
parser.add_argument('--rag-weight', default=0.05, type=float)
parser.add_argument('--rag-train-list', default='', type=str)
parser.add_argument('--rag-conf-gate', action='store_true')
# DSANet modules
parser.add_argument('--use-dnp', action='store_true')
parser.add_argument('--dnp-num-prototypes', default=16, type=int)
parser.add_argument('--dnp-decoder-depth', default=8, type=int)
parser.add_argument('--dnp-normal-selection-ratio', default=0.8, type=float)
parser.add_argument('--use-clip-adapter', action='store_true')
parser.add_argument('--clip-adapter-layers', default=3, type=int)
parser.add_argument('--clip-adapter-weight', default=0.1, type=float)
parser.add_argument('--use-dcsa', action='store_true')
parser.add_argument('--dcsa-loss-weight', default=0.3, type=float)
parser.add_argument('--dnp-loss-weight', default=0.05, type=float)
parser.add_argument('--joint-student-visual-only', action='store_true',
                    help='Force student in joint stage to take visual-only path (no audio/snippet/rag) to prevent multimodal shortcut')
parser.add_argument('--loss2-weight', default=1.0, type=float)
parser.add_argument('--temp', default=5.0, type=float)
parser.add_argument('--kd-temp', default=2.0, type=float, help='KD softmax temperature for kd_multi')
parser.add_argument('--logits3-alpha', default=0.0, type=float,
                    help='When use_dcsa, fuse logits3 into AP2/MAP scoring: softmax(logits2 + alpha*logits3)')
parser.add_argument('--teacher-epochs', default=3, type=int)
parser.add_argument('--teacher-lr', default=1e-5, type=float)
parser.add_argument('--teacher-scheduler-milestones', default=[2, 4], nargs='+', type=int)
parser.add_argument('--teacher-model-path', default='model/model_xd_teacher.pth', type=str)
parser.add_argument('--train-teacher', action='store_true')
parser.add_argument('--distill-weight', default=0.5, type=float)
parser.add_argument('--distill-kd-bin-weight', default=0.35, type=float)
parser.add_argument('--distill-kd-multi-weight', default=0.45, type=float)
parser.add_argument('--distill-kd-feat-weight', default=0.2, type=float)
parser.add_argument('--acc-dense-distill-weight', default=0.0, type=float)
parser.add_argument('--acc-eta', default=0.5, type=float)
parser.add_argument('--acc-threshold', default=0.9, type=float)
parser.add_argument('--loss1-weight', default=0.7, type=float)
parser.add_argument('--event-kd-weight', default=0.3, type=float)
parser.add_argument('--event-trend-weight', default=0.15, type=float)
parser.add_argument('--pretrained-path', default='model/model_xd.pth', type=str)
parser.add_argument('--log-path', default='model/xd_teacher_student.log', type=str)
parser.add_argument('--train-stage', default='joint', choices=['joint', 'stage1', 'stage2', 'two_stage'])
parser.add_argument('--stage1-kd-source', default='teacher_visual_only_outputs', choices=['teacher_visual_only_outputs', 'teacher_current_outputs'])
parser.add_argument('--stage1-teacher-use-audio', action='store_true')
parser.add_argument('--stage1-teacher-use-snippet-gating', action='store_true')
parser.add_argument('--stage1-teacher-use-rag', action='store_true')
parser.add_argument('--stage1-student-force-visual-only', action='store_true')
parser.add_argument('--stage1-student-classification-on-pure-visual', action='store_true')
parser.add_argument('--stage1-teacher-use-causal-graph', action='store_true')
parser.add_argument('--stage1-student-use-causal-graph', action='store_true')
parser.add_argument('--stage2-init-model-path', default='', type=str)
parser.add_argument('--stage2-model-path', default='', type=str)
parser.add_argument('--stage2-checkpoint-path', default='', type=str)
parser.add_argument('--stage2-log-path', default='', type=str)
parser.add_argument('--stage2-use-audio', action='store_true')
parser.add_argument('--stage2-use-snippet-gating', action='store_true')
parser.add_argument('--stage2-use-rag', action='store_true')
parser.add_argument('--stage2-use-causal-graph', action='store_true')
parser.add_argument('--stage2-distill-weight', default=0.0, type=float)


_XD_TEXT_DIR_MAP = {
    'v1': (
        'data/xd_text',
        'data/xd_text_llm_snippets',
    ),
    'xd_text': (
        'data/xd_text',
        'data/xd_text_llm_snippets',
    ),
    'v2': (
        'data/xd_text_v2',
        'data/xd_text_v2_snippets',
    ),
    'xd_text_v2': (
        'data/xd_text_v2',
        'data/xd_text_v2_snippets',
    ),
}


def resolve_text_feature_dirs(args):
    version = getattr(args, 'text_feature_version', 'v1')
    default_text_dir, default_snippet_dir = _XD_TEXT_DIR_MAP[version]
    if not getattr(args, 'text_feature_dir', None):
        args.text_feature_dir = default_text_dir
    if not getattr(args, 'snippet_text_feature_dir', None):
        args.snippet_text_feature_dir = default_snippet_dir
    args.text_feature_dir = str(Path(args.text_feature_dir))
    args.snippet_text_feature_dir = str(Path(args.snippet_text_feature_dir))

    if args.train_stage in {'stage1', 'two_stage'}:
        if not args.stage1_student_force_visual_only:
            args.stage1_student_force_visual_only = True
        if not args.stage1_student_classification_on_pure_visual:
            args.stage1_student_classification_on_pure_visual = True
    if not args.stage2_model_path:
        args.stage2_model_path = args.model_path
    if not args.stage2_checkpoint_path:
        args.stage2_checkpoint_path = args.checkpoint_path
    if not args.stage2_log_path:
        args.stage2_log_path = args.log_path
    return args
