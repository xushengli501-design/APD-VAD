import time as _time
import torch
import contextlib
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.ndimage import gaussian_filter1d

from model_ucf import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.ucf_detectionMAP import getDetectionMAP as dmAP
import ucf_option


def _safe_torch_load(path, map_location='cpu'):
    with contextlib.suppress(TypeError):
        return torch.load(path, map_location=map_location, weights_only=True)
    return torch.load(path, map_location=map_location)


def _extract_state_dict(payload):
    if isinstance(payload, dict) and isinstance(payload.get('model_state_dict'), dict):
        return payload['model_state_dict']
    if isinstance(payload, dict) and payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
        return payload
    return None


def _split_snippet_text(snippet_text, lengths, maxlen):
    if snippet_text is None:
        return None
    if snippet_text.dim() == 3:
        return snippet_text
    chunks = []
    total_len = snippet_text.shape[0]
    offset = 0
    for length in lengths:
        cur_len = int(length.item())
        if offset >= total_len:
            chunk = torch.zeros(maxlen, snippet_text.shape[-1], device=snippet_text.device, dtype=snippet_text.dtype)
        else:
            chunk = snippet_text[offset:min(offset + cur_len, total_len)]
            if chunk.shape[0] < maxlen:
                pad = torch.zeros(maxlen - chunk.shape[0], snippet_text.shape[-1], device=snippet_text.device, dtype=snippet_text.dtype)
                chunk = torch.cat([chunk, pad], dim=0)
        offset += cur_len
        chunks.append(chunk[:maxlen])
    return torch.stack(chunks, dim=0)


def _smooth_1d_scores(scores, sigma):
    if sigma is None or sigma <= 0:
        return scores
    return gaussian_filter1d(scores, sigma=float(sigma), mode='nearest')


def _smooth_2d_scores(scores, sigma):
    if sigma is None or sigma <= 0:
        return scores
    smoothed = gaussian_filter1d(scores, sigma=float(sigma), axis=0, mode='nearest')
    smoothed = np.clip(smoothed, 1e-8, None)
    smoothed = smoothed / np.clip(smoothed.sum(axis=1, keepdims=True), 1e-8, None)
    return smoothed


def refine_scores_hierarchical(logits_mlp, logits_align, temp=5.0):
    epsilon = 1e-12
    total_abnormal_prob = torch.sigmoid(logits_mlp / temp)
    total_normal_prob = 1.0 - total_abnormal_prob
    p_align = F.softmax(logits_align / temp, dim=1)
    p_align_abnormal_only = p_align[:, 1:]
    sum_p_align_abnormal = p_align_abnormal_only.sum(dim=1, keepdim=True)
    abnormal_distribution = p_align_abnormal_only / (sum_p_align_abnormal + epsilon)
    final_abnormal_probs = total_abnormal_prob * abnormal_distribution
    final_probabilities = torch.cat([total_normal_prob, final_abnormal_probs], dim=1)
    return final_probabilities


def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device, use_snippet_text_gating=False,
         use_motion_refine=False, use_category_refine=False, prototype_temp=0.07, gaussian_sigma=5.0,
         return_map=False, DNP_use=True, temp=5.0):

    model.to(device)
    model.eval()

    element_logits2_stack = []
    _inf_times = []
    _total_frames = 0
    _total_snippets = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            _t0 = _time.perf_counter()
            visual = item[0].squeeze(0)
            length = item[2]
            snippet_text = item[3].squeeze(0).to(device) if use_snippet_text_gating else None

            length = int(length)
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)
                if snippet_text is not None:
                    snippet_text = snippet_text.unsqueeze(0)

            visual = visual.to(device)

            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            if snippet_text is not None:
                snippet_text = _split_snippet_text(snippet_text, lengths, maxlen)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            result = model(
                visual, padding_mask, prompt_text, lengths,
                DNP_use=DNP_use,
                snippet_text_features=snippet_text,
                use_motion_refine=use_motion_refine,
                use_category_refine=use_category_refine,
                prototype_temp=prototype_temp,
            )
            # result = (text_features_ori, logits1, logits2, logits3, logits4[, ...])
            _, logits1, logits2 = result[0], result[1], result[2]
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            optimized_probs = refine_scores_hierarchical(logits1[0:len_cur], logits2[0:len_cur], temp)
            prob2 = (1 - optimized_probs[:, 0]).detach().cpu().numpy()
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1)).detach().cpu().numpy()
            prob1 = _smooth_1d_scores(prob1, gaussian_sigma)
            prob2 = _smooth_1d_scores(prob2, gaussian_sigma)
            prob1 = torch.from_numpy(prob1)
            prob2 = torch.from_numpy(prob2)

            if i == 0:
                ap1 = prob1
                ap2 = prob2
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            element_logits2 = optimized_probs.detach().cpu().numpy()
            element_logits2 = _smooth_2d_scores(element_logits2, gaussian_sigma)
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)
            _inf_times.append(_time.perf_counter() - _t0)
            _total_frames += 16 * len_cur
            _total_snippets += len_cur

    if _inf_times:
        _n = len(_inf_times)
        _total_time = sum(_inf_times)
        _mean_ms = 1000.0 * _total_time / _n
        _max_ms = 1000.0 * max(_inf_times)
        _feat_sps = _total_snippets / _total_time if _total_time > 0 else 0.0
        _frame_fps = _total_frames / _total_time if _total_time > 0 else 0.0
        print(
            f'[prof] Inference Time / Video: {_mean_ms:.1f} ms (mean over {_n} videos, '
            f'wall-clock; max {_max_ms:.1f} ms) — '
            f'FEATURE-LEVEL: pre-extracted CLIP feature -> anomaly scores, '
            f'excludes video decode + CLIP visual encoding + feature loading',
            flush=True,
        )
        print(
            f'[prof] Feature-level throughput: {_feat_sps:.0f} snippets/s '
            f'({_total_snippets} snippets in {_total_time:.1f} s)',
            flush=True,
        )
        print(
            f'[prof] Frame-equivalent FPS (16 frames/snippet, feature-level only, '
            f'NOT end-to-end): {_frame_fps:.0f} FPS '
            f'({_total_frames} frames = 16 x {_total_snippets} snippets in {_total_time:.1f} s)',
            flush=True,
        )
        if torch.cuda.is_available():
            _peak_alloc = torch.cuda.max_memory_allocated(device) / 1e9
            _peak_resv = torch.cuda.max_memory_reserved(device) / 1e9
            print(
                f'[prof] torch.cuda in-process peak: {_peak_alloc:.2f} GB allocated '
                f'({_peak_resv:.2f} GB reserved) — cross-check vs nvidia-smi peak',
                flush=True,
            )

    ap1 = ap1.cpu().numpy()
    ap2 = ap2.cpu().numpy()
    ap1 = ap1.tolist()
    ap2 = ap2.tolist()

    pred1 = np.repeat(ap1, 16)
    pred2 = np.repeat(ap2, 16)
    gt_vec = np.asarray(gt)
    if len(pred1) != len(gt_vec):
        n = min(len(pred1), len(gt_vec))
        print(f"[warn] score/gt length mismatch: pred={len(pred1)}, gt={len(gt_vec)}; truncate to {n}", flush=True)
        pred1 = pred1[:n]
        pred2 = pred2[:n]
        gt_vec = gt_vec[:n]

    ROC1 = roc_auc_score(gt_vec, pred1)
    AP1 = average_precision_score(gt_vec, pred1)
    ROC2 = roc_auc_score(gt_vec, pred2)
    AP2 = average_precision_score(gt_vec, pred2)

    print("AUC1: ", ROC1, " AP1: ", AP1)
    print("AUC2: ", ROC2, " AP2:", AP2)

    n_vid = min(len(element_logits2_stack), len(gtsegments), len(gtlabels))
    if n_vid != len(element_logits2_stack):
        print(
            f"[warn] proposal/gt video count mismatch: pred={len(element_logits2_stack)}, gt={len(gtsegments)}; truncate to {n_vid}",
            flush=True,
        )
    dmap, iou = dmAP(
        element_logits2_stack[:n_vid],
        gtsegments[:n_vid],
        gtlabels[:n_vid],
        excludeNormal=False,
    )
    averageMAP = 0
    for i in range(5):
        print('mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap[i]))
        averageMAP += dmap[i]
    averageMAP = averageMAP / (i + 1)
    print('average MAP: {:.2f}'.format(averageMAP))

    if return_map:
        return ROC1, AP1, averageMAP
    return ROC1, AP1


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()
    args = ucf_option.resolve_text_feature_dirs(args)
    print(f'Text feature version: {args.text_feature_version}')
    print(f'Text feature dir: {args.text_feature_dir}')
    print(f'Snippet text feature dir: {args.snippet_text_feature_dir}')

    label_map = dict({'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson', 'Assault': 'Assault', 'Burglary': 'Burglary', 'Explosion': 'Explosion', 'Fighting': 'Fighting', 'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery', 'Shooting': 'Shooting', 'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing', 'Vandalism': 'Vandalism'})

    testdataset = UCFDataset(
        args.visual_length, args.test_list, True, label_map,
        snippet_text_feature_dir=args.snippet_text_feature_dir,
        return_snippet_text_feature=args.use_snippet_text_gating,
    )
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = CLIPVAD(
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
        args.visual_head, args.visual_layers, args.attn_window,
        args.prompt_prefix, args.prompt_postfix, device,
        dataset='ucf',
        use_debiased_causal_graph=args.use_debiased_causal_graph,
        debiased_graph_threshold=args.debiased_graph_threshold,
        text_adapt_until=getattr(args, 'text_adapt_until', 3),
        t_w=getattr(args, 't_w', 0.1),
        num_prototypes=getattr(args, 'num_prototypes', 16),
        decoder_depth=getattr(args, 'decoder_depth', 8),
        normal_selection_ratio=getattr(args, 'normal_selection_ratio', 0.8),
    )

    if args.model_bundle_path:
        bundle = _safe_torch_load(args.model_bundle_path, map_location='cpu')
        if not isinstance(bundle, dict):
            raise ValueError(f'Invalid bundle format: {args.model_bundle_path}')

        # Support delta bundle: reconstruct best_map from best_auc1 + delta
        base_sd = bundle.get('best_auc1_state_dict')
        map_sd = bundle.get('best_map_state_dict')
        if map_sd is None and isinstance(base_sd, dict) and isinstance(bundle.get('delta_to_best_map'), dict):
            delta = bundle['delta_to_best_map']
            map_sd = {k: base_sd[k] + delta[k] if k in delta else base_sd[k] for k in base_sd}

        candidates = [
            ('best_auc1', base_sd),
            ('best_map', map_sd),
        ]

        metrics = {}
        for tag, state_dict in candidates:
            if not isinstance(state_dict, dict):
                print(f'[warn] Missing {tag} weights in bundle: {args.model_bundle_path}', flush=True)
                continue
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(
                f'===== evaluating {tag} from bundle ===== '
                f'(missing={len(missing)}, unexpected={len(unexpected)})',
                flush=True,
            )
            auc1, ap1, avg_map = test(
                model, testdataloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device,
                use_snippet_text_gating=args.use_snippet_text_gating,
                use_motion_refine=args.use_motion_refine,
                use_category_refine=args.use_category_refine,
                prototype_temp=args.prototype_temp,
                gaussian_sigma=args.gaussian_sigma,
                DNP_use=getattr(args, 'DNP_use', True),
                temp=getattr(args, 'temp', 5.0),
                return_map=True,
            )
            metrics[tag] = {'auc1': float(auc1), 'ap1': float(ap1), 'map': float(avg_map)}

        if metrics:
            best_auc = max(v['auc1'] for v in metrics.values())
            best_map = max(v['map'] for v in metrics.values())
            print('===== bundle summary =====', flush=True)
            for tag, result in metrics.items():
                print(
                    f'{tag}: AUC1={result["auc1"]:.6f}, AP1={result["ap1"]:.6f}, mAP={result["map"]:.6f}',
                    flush=True,
                )
            print(f'best AUC1 = {best_auc:.6f}', flush=True)
            print(f'best mAP  = {best_map:.6f}', flush=True)
    else:
        model_payload = _safe_torch_load(args.model_path, map_location='cpu')
        model_state = _extract_state_dict(model_payload)
        if model_state is None:
            raise ValueError(f'Unsupported model format: {args.model_path}')
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        print(f'Loaded model with strict=False. Missing: {len(missing)}, Unexpected: {len(unexpected)}', flush=True)

        test(
            model, testdataloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device,
            use_snippet_text_gating=args.use_snippet_text_gating,
            use_motion_refine=args.use_motion_refine,
            use_category_refine=args.use_category_refine,
            prototype_temp=args.prototype_temp,
            gaussian_sigma=args.gaussian_sigma,
            DNP_use=getattr(args, 'DNP_use', True),
            temp=getattr(args, 'temp', 5.0),
        )
