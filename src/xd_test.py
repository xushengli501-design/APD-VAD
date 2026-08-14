import time as _time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import contextlib
from sklearn.metrics import average_precision_score, roc_auc_score

from model_xd import CLIPVAD
from utils.dataset import XDDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.xd_detectionMAP import getDetectionMAP as dmAP
import xd_option


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



def refine_scores_hierarchical(logits_mlp, logits_align, temp=1.0):
    total_abnormal_prob = torch.sigmoid(logits_mlp / temp)
    total_normal_prob = 1.0 - total_abnormal_prob
    p_align = F.softmax(logits_align / temp, dim=1)
    p_align_abn = p_align[:, 1:]
    abn_dist = p_align_abn / p_align_abn.sum(dim=1, keepdim=True).clamp_min(1e-12)
    final_abn = total_abnormal_prob * abn_dist
    return torch.cat([total_normal_prob, final_abn], dim=1)


def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device,
         use_motion_refine=False, use_audio_aux=False, use_snippet_text_gating=False, frame_offsets=None,
         use_rag=False, force_visual_only=False, classification_on_pure_visual=False,
         override_use_debiased_causal_graph=None, use_dcsa=False, temp=1.0, logits3_alpha=0.0):

    model.to(device)
    model.eval()

    element_logits2_stack = []
    kept_indices = []
    _inf_times = []
    _total_frames = 0
    _total_snippets = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            _t0 = _time.perf_counter()
            cursor = 0
            visual = item[cursor].squeeze(0)
            cursor += 1
            audio = None
            if use_audio_aux:
                audio = item[cursor].squeeze(0)
                cursor += 1
            length = item[cursor + 1]
            cursor += 2
            snippet_text = item[cursor].squeeze(0).to(device) if use_snippet_text_gating else None
            if use_snippet_text_gating:
                cursor += 1
            rag_features = None
            if use_rag:
                rag_class_prior = item[cursor].to(device)
                rag_anomaly_score = item[cursor + 1].to(device)
                rag_confidence = item[cursor + 2].to(device)
                cursor += 3
                rag_features = torch.cat([rag_class_prior, rag_anomaly_score, rag_confidence], dim=-1)
            sample_index = int(item[-1]) if use_audio_aux else i

            kept_indices.append(sample_index)
            length = int(length)
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)
                if audio is not None and audio.dim() == 2:
                    audio = audio.unsqueeze(0)
                if snippet_text is not None:
                    snippet_text = snippet_text.unsqueeze(0)

            visual = visual.to(device)
            if audio is not None:
                audio = audio.to(device)

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
            fwd_out = model(
                visual, padding_mask, prompt_text, lengths,
                use_motion_refine=use_motion_refine,
                audio=audio,
                use_audio_aux=use_audio_aux,
                snippet_text_features=snippet_text,
                rag_features=rag_features,
                force_visual_only=force_visual_only,
                classification_on_pure_visual=classification_on_pure_visual,
                override_use_debiased_causal_graph=override_use_debiased_causal_graph,
                use_dcsa=use_dcsa,
            )
            logits1 = fwd_out[1]
            logits2 = fwd_out[2]
            if use_dcsa and logits3_alpha != 0.0:
                logits3 = fwd_out[3]
                # logits2: (B,T,C); logits3: (B,1,C) — broadcast event-centric prior to all snippets
                logits2 = logits2 + logits3_alpha * logits3
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])

            if use_dcsa:
                optimized_probs = refine_scores_hierarchical(logits1[0:len_cur], logits2[0:len_cur], temp)
                prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
                element_logits2_out = optimized_probs.detach().cpu().numpy()
            else:
                prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
                element_logits2_out = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            if i == 0:
                ap1 = prob1
                ap2 = prob2
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            element_logits2_out = np.repeat(element_logits2_out, 16, 0)
            element_logits2_stack.append(element_logits2_out)
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

    ap1 = ap1.cpu().numpy().tolist()
    ap2 = ap2.cpu().numpy().tolist()

    gt_eval = gt
    gtsegments_eval = gtsegments
    gtlabels_eval = gtlabels
    if use_audio_aux:
        kept_indices = np.array(kept_indices, dtype=np.int64)
        gtsegments_eval = gtsegments[kept_indices]
        gtlabels_eval = gtlabels[kept_indices]
        frame_counts = np.array([len(pred) for pred in element_logits2_stack], dtype=np.int64)
        gt_mask_parts = []
        for sample_index, frame_count in zip(kept_indices, frame_counts):
            start = int(frame_offsets[sample_index])
            gt_mask_parts.append(np.arange(start, start + frame_count))
        gt_eval = gt[np.concatenate(gt_mask_parts)]

    ROC1 = roc_auc_score(gt_eval, np.repeat(ap1, 16))
    AP1 = average_precision_score(gt_eval, np.repeat(ap1, 16))
    ROC2 = roc_auc_score(gt_eval, np.repeat(ap2, 16))
    AP2 = average_precision_score(gt_eval, np.repeat(ap2, 16))

    print("AUC1: ", ROC1, " AP1: ", AP1)
    print("AUC2: ", ROC2, " AP2:", AP2)

    dmap, iou = dmAP(element_logits2_stack, gtsegments_eval, gtlabels_eval, excludeNormal=False)
    averageMAP = 0
    for i in range(5):
        print('mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap[i]))
        averageMAP += dmap[i]
    averageMAP = averageMAP/(i+1)
    print('average MAP: {:.2f}'.format(averageMAP))

    return ROC1, AP2 ,averageMAP


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()
    args = xd_option.resolve_text_feature_dirs(args)

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

    test_dataset = XDDataset(
        args.visual_length, args.test_list, True, label_map,
        use_audio=args.use_audio_aux, audio_root=args.audio_root,
        snippet_text_feature_dir=args.snippet_text_feature_dir,
        return_snippet_text_feature=args.use_snippet_text_gating,
        use_rag=args.use_rag,
        rag_topk=args.rag_topk,
        rag_max_bank_size=args.rag_max_bank_size,
        rag_train_list=args.rag_train_list or args.train_list,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    frame_offsets = None
    if args.use_audio_aux:
        full_test_df = pd.read_csv(args.test_list)
        frame_counts = full_test_df['path'].map(lambda p: np.load(p).shape[0] * 16).to_numpy(dtype=np.int64)
        frame_offsets = np.concatenate(([0], np.cumsum(frame_counts[:-1], dtype=np.int64)))

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = CLIPVAD(
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
        args.visual_head, args.visual_layers, args.attn_window,
        args.prompt_prefix, args.prompt_postfix, device,
        audio_dim=args.audio_dim,
        audio_fusion_mode=args.audio_fusion_mode,
        audio_cross_attn_heads=args.audio_cross_attn_heads,
        use_debiased_causal_graph=args.use_debiased_causal_graph,
        debiased_graph_threshold=args.debiased_graph_threshold,
        causal_repr_alpha=args.causal_repr_alpha,
        causal_repr_detach=args.causal_repr_detach,
        snippet_gate_temperature=args.snippet_gate_temperature,
        snippet_gate_residual=args.snippet_gate_residual,
        use_rag=args.use_rag,
        rag_weight=args.rag_weight,
        rag_conf_gate=args.rag_conf_gate,
        use_dnp=getattr(args, 'use_dnp', False),
        dnp_num_prototypes=getattr(args, 'dnp_num_prototypes', 16),
        dnp_decoder_depth=getattr(args, 'dnp_decoder_depth', 8),
        dnp_normal_selection_ratio=getattr(args, 'dnp_normal_selection_ratio', 0.8),
        use_clip_adapter=getattr(args, 'use_clip_adapter', False),
        clip_adapter_layers=getattr(args, 'clip_adapter_layers', 3),
        clip_adapter_weight=getattr(args, 'clip_adapter_weight', 0.1),
    )
    model_payload = _safe_torch_load(args.model_path, map_location='cpu')
    model_state = _extract_state_dict(model_payload)
    if model_state is None:
        raise ValueError(f'Unsupported model format: {args.model_path}')
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(f'Loaded model with strict=False. Missing: {len(missing)}, Unexpected: {len(unexpected)}', flush=True)

    test(
        model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device,
        use_motion_refine=args.use_motion_refine,
        use_audio_aux=args.use_audio_aux,
        use_snippet_text_gating=args.use_snippet_text_gating,
        frame_offsets=frame_offsets,
        use_rag=args.use_rag,
        force_visual_only=args.train_stage == 'stage1' and args.stage1_student_force_visual_only,
        classification_on_pure_visual=args.train_stage == 'stage1' and args.stage1_student_classification_on_pure_visual,
        override_use_debiased_causal_graph=(args.stage1_student_use_causal_graph if args.train_stage == 'stage1' else None),
        use_dcsa=getattr(args, 'use_dcsa', False),
        temp=args.temp,
        logits3_alpha=getattr(args, 'logits3_alpha', 0.0),
    )
