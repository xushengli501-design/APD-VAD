import os
import contextlib
import sys
from datetime import datetime
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler
import numpy as np
import pandas as pd
import random

from model_xd import CLIPVAD
from xd_test import test
from utils.dataset import XDDataset
from utils.tools import get_prompt_text, get_batch_label
from utils.StableAdamW import StableAdamW
import xd_option


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()



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



def _load_pretrained_if_available(model, pretrained_path: str):
    if not pretrained_path:
        return
    if not os.path.exists(pretrained_path):
        print(f'Pretrained not found: {pretrained_path}', flush=True)
        return
    print(f'Loading pretrained: {pretrained_path}', flush=True)
    payload = _safe_torch_load(pretrained_path, map_location='cpu')
    state_dict = _extract_state_dict(payload)
    if state_dict is None:
        raise ValueError(f'Unsupported model format: {pretrained_path}')
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'Loaded with strict=False. Missing: {len(missing)}, Unexpected: {len(unexpected)}', flush=True)



def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    return -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)



def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp).view(1)], dim=0)
    return F.binary_cross_entropy(instance_logits, labels)



def text_contrast_loss(text_features):
    if text_features.dim() == 3:
        text_features = text_features.mean(dim=0)
    loss = torch.zeros(1, device=text_features.device)
    text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
    for j in range(1, text_features.shape[0]):
        text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
        loss += torch.abs(text_feature_normal @ text_feature_abr)
    return loss / 6



def temporal_consistency_loss(logits, lengths):
    scores = torch.sigmoid(logits.squeeze(-1))
    total_loss = scores.new_tensor(0.0)
    valid_batches = 0
    for i in range(scores.shape[0]):
        valid_length = int(lengths[i].item())
        if valid_length <= 1:
            continue
        score_seq = scores[i, :valid_length]
        smooth_loss = torch.mean((score_seq[1:] - score_seq[:-1]) ** 2)
        local_avg = F.avg_pool1d(score_seq.view(1, 1, -1), kernel_size=3, stride=1, padding=1).view(-1)
        event_loss = torch.mean((score_seq - local_avg) ** 2)
        total_loss = total_loss + smooth_loss + event_loss
        valid_batches += 1
    if valid_batches == 0:
        return total_loss
    return total_loss / valid_batches



def CLASM_EVENT(logits, labels, lengths, device, epsilon=0.1):
    num_classes = logits.shape[2]
    instance_logits = torch.zeros(0).to(device)
    labels_sum = labels.sum(dim=1, keepdim=True).clamp(min=1e-6)
    labels_sm = (1 - epsilon) * (labels / labels_sum) + epsilon / num_classes
    labels_sm = labels_sm.to(device)
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=1, largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    return -torch.mean(torch.sum(labels_sm * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)



def CLASM_BKG(logits, labels, lengths, device, epsilon=0.1):
    num_classes = logits.shape[2]
    instance_logits = torch.zeros(0).to(device)
    labels2 = torch.full(labels.shape, 0.01, device=labels.device)
    labels2[:, 0] = 1
    labels2_sum = labels2.sum(dim=1, keepdim=True).clamp(min=1e-6)
    labels2 = (1 - epsilon) * (labels2 / labels2_sum) + epsilon / num_classes
    labels2 = labels2.to(device)
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=1, largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    return -torch.mean(torch.sum(labels2 * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)



class WarmCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for _ in self.base_lrs]
        return [self.schedule[self.last_epoch] for _ in self.base_lrs]



class ConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits1, original_features, reconstructed_features, lengths):
        recon_error = (1.0 - F.cosine_similarity(original_features, reconstructed_features, dim=-1)) / 2.0
        classifier_prob = torch.sigmoid(logits1.squeeze(-1))
        mask = torch.arange(logits1.shape[1], device=logits1.device)[None, :] < lengths[:, None]
        return F.mse_loss(classifier_prob[mask], recon_error[mask])


_consistency_loss_fn = ConsistencyLoss()



def event_completeness_distillation(student_logits, teacher_logits, lengths):
    student_scores = torch.sigmoid(student_logits.squeeze(-1))
    teacher_scores = torch.sigmoid(teacher_logits.squeeze(-1))
    shape_loss = student_scores.new_tensor(0.0)
    trend_loss = student_scores.new_tensor(0.0)
    valid_batches = 0
    for i in range(student_scores.shape[0]):
        valid_length = int(lengths[i].item())
        if valid_length <= 1:
            continue
        student_seq = student_scores[i, :valid_length]
        teacher_seq = teacher_scores[i, :valid_length]
        shape_loss = shape_loss + F.mse_loss(student_seq, teacher_seq)
        if valid_length > 2:
            student_diff = student_seq[1:] - student_seq[:-1]
            teacher_diff = teacher_seq[1:] - teacher_seq[:-1]
            trend_loss = trend_loss + F.mse_loss(student_diff, teacher_diff)
        valid_batches += 1
    if valid_batches == 0:
        return shape_loss, trend_loss
    return shape_loss / valid_batches, trend_loss / valid_batches



def build_teacher_text_features(model, prompt_text, text_feature_batch, device):
    class_text = model.encode_textprompt(prompt_text)
    class_text = F.normalize(class_text, dim=-1)
    caption_text = text_feature_batch.to(device=device, dtype=class_text.dtype)
    caption_text = F.normalize(caption_text, dim=-1)
    normal_anchor = class_text[0].unsqueeze(0)
    teacher_text = class_text.unsqueeze(0) + caption_text.unsqueeze(1) * normal_anchor.unsqueeze(1)
    return F.normalize(teacher_text, dim=-1)



def _teacher_best_path(teacher_model_path: str) -> str:
    if teacher_model_path.endswith('.pth'):
        return teacher_model_path[:-4] + '_best.pth'
    return teacher_model_path + '_best'


def _suffix_path(path: str, suffix: str) -> str:
    if path.endswith('.pth'):
        return path[:-4] + suffix + '.pth'
    return path + suffix



def _setup_logging(log_path: str):
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    log_file = open(log_path, 'a', buffering=1)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    print(f"\n===== XD Teacher-Student training started at {datetime.now().isoformat(timespec='seconds')} =====", flush=True)
    print(f'Logging to {log_path}', flush=True)
    return log_file



def _try_restore_best(model, checkpoint_path: str):
    if not os.path.isfile(checkpoint_path):
        return False
    checkpoint = _safe_torch_load(checkpoint_path)
    state_dict = checkpoint.get('model_state_dict') if isinstance(checkpoint, dict) else None
    if state_dict is None:
        return False
    try:
        model.load_state_dict(state_dict, strict=False)
        return True
    except RuntimeError as exc:
        print(f'Skip incompatible checkpoint {checkpoint_path}: {exc}', flush=True)
        return False



def _unpack_xd_ts_batch(batch, use_audio_aux=False, use_text_feature=True, use_snippet_text_gating=False, use_rag=False):
    cursor = 0
    visual_feat = batch[cursor]
    cursor += 1
    audio_feat = None
    if use_audio_aux:
        audio_feat = batch[cursor]
        cursor += 1
    clip_labels = batch[cursor]
    feat_lengths = batch[cursor + 1]
    cursor += 2
    text_features = None
    if use_text_feature:
        text_features = batch[cursor]
        cursor += 1
    snippet_text = None
    if use_snippet_text_gating:
        snippet_text = batch[cursor]
        cursor += 1
    rag_class_prior = None
    rag_anomaly_score = None
    rag_confidence = None
    if use_rag:
        rag_class_prior = batch[cursor]
        rag_anomaly_score = batch[cursor + 1]
        rag_confidence = batch[cursor + 2]
        cursor += 3
    metadata = batch[cursor] if cursor < len(batch) and not use_audio_aux else None
    original_index = batch[-1] if use_audio_aux else None
    return (
        visual_feat, audio_feat, clip_labels, feat_lengths, text_features,
        snippet_text, rag_class_prior, rag_anomaly_score, rag_confidence,
        metadata, original_index,
    )



def _build_rag_features(rag_class_prior, rag_anomaly_score, rag_confidence, device, dtype):
    if rag_class_prior is None:
        return None
    return torch.cat([
        rag_class_prior.to(device=device, dtype=dtype),
        rag_anomaly_score.to(device=device, dtype=dtype),
        rag_confidence.to(device=device, dtype=dtype),
    ], dim=-1)



def _stage1_teacher_use_audio(args):
    if args.train_stage in {'stage1', 'two_stage'}:
        return args.stage1_teacher_use_audio
    return args.use_audio_aux



def _stage1_teacher_use_snippet(args):
    if args.train_stage in {'stage1', 'two_stage'}:
        return args.stage1_teacher_use_snippet_gating
    return args.use_snippet_text_gating



def _stage1_teacher_use_rag(args):
    if args.train_stage in {'stage1', 'two_stage'}:
        return args.stage1_teacher_use_rag
    return args.use_rag



def _build_teacher_forward_kwargs(args, stage, audio_batch=None, snippet_text_batch=None, rag_features=None, kd_pass=False):
    if stage == 'joint':
        # Teacher uses visual + text guidance only; audio/snippet/RAG are student-side privileges
        return {
            'snippet_text_features': None,
            'use_motion_refine': args.use_motion_refine,
            'audio': None,
            'use_audio_aux': False,
            'rag_features': None,
        }

    if stage == 'stage1':
        force_visual_only = kd_pass and args.stage1_kd_source == 'teacher_visual_only_outputs'
        return {
            'snippet_text_features': snippet_text_batch if _stage1_teacher_use_snippet(args) and not force_visual_only else None,
            'use_motion_refine': args.use_motion_refine,
            'audio': audio_batch if _stage1_teacher_use_audio(args) and not force_visual_only else None,
            'use_audio_aux': _stage1_teacher_use_audio(args) and not force_visual_only,
            'rag_features': rag_features if _stage1_teacher_use_rag(args) and not force_visual_only else None,
            'force_visual_only': force_visual_only,
            'classification_on_pure_visual': force_visual_only,
            'override_use_debiased_causal_graph': args.stage1_teacher_use_causal_graph,
        }

    raise ValueError(f'Unsupported teacher stage: {stage}')



def _build_student_forward_kwargs(args, stage, audio_batch=None, snippet_text_batch=None, rag_features=None):
    if stage == 'joint':
        # Student uses all modalities except text override (teacher's privilege)
        return {
            'snippet_text_features': snippet_text_batch if args.use_snippet_text_gating else None,
            'use_motion_refine': args.use_motion_refine,
            'audio': audio_batch,
            'use_audio_aux': args.use_audio_aux,
            'rag_features': rag_features,
        }

    if stage == 'stage1':
        return {
            'snippet_text_features': None,
            'use_motion_refine': args.use_motion_refine,
            'audio': None,
            'use_audio_aux': False,
            'rag_features': None,
            'force_visual_only': args.stage1_student_force_visual_only,
            'classification_on_pure_visual': args.stage1_student_classification_on_pure_visual,
            'override_use_debiased_causal_graph': args.stage1_student_use_causal_graph,
        }

    if stage == 'stage2':
        return {
            'snippet_text_features': snippet_text_batch if args.stage2_use_snippet_gating else None,
            'use_motion_refine': args.use_motion_refine,
            'audio': audio_batch if args.stage2_use_audio else None,
            'use_audio_aux': args.stage2_use_audio,
            'rag_features': rag_features if args.stage2_use_rag else None,
            'override_use_debiased_causal_graph': args.stage2_use_causal_graph,
        }

    raise ValueError(f'Unsupported student stage: {stage}')



def _build_dataset(stage, args, label_map, is_test):
    if stage == 'joint':
        return XDDataset(
            args.visual_length,
            args.test_list if is_test else args.train_list,
            is_test,
            label_map,
            use_audio=args.use_audio_aux,
            audio_root=args.audio_root,
            text_feature_dir=args.text_feature_dir if not is_test else None,
            return_text_feature=not is_test,
            snippet_text_feature_dir=args.snippet_text_feature_dir,
            return_snippet_text_feature=args.use_snippet_text_gating,
            use_rag=args.use_rag,
            rag_topk=args.rag_topk,
            rag_max_bank_size=args.rag_max_bank_size,
            rag_train_list=args.rag_train_list or args.train_list,
        )

    if stage == 'stage1':
        teacher_uses_audio = _stage1_teacher_use_audio(args)
        teacher_uses_snippet = _stage1_teacher_use_snippet(args)
        teacher_uses_rag = _stage1_teacher_use_rag(args)
        return XDDataset(
            args.visual_length,
            args.test_list if is_test else args.train_list,
            is_test,
            label_map,
            use_audio=teacher_uses_audio,
            audio_root=args.audio_root,
            text_feature_dir=args.text_feature_dir if not is_test else None,
            return_text_feature=not is_test,
            snippet_text_feature_dir=args.snippet_text_feature_dir,
            return_snippet_text_feature=teacher_uses_snippet,
            use_rag=teacher_uses_rag,
            rag_topk=args.rag_topk,
            rag_max_bank_size=args.rag_max_bank_size,
            rag_train_list=args.rag_train_list or args.train_list,
        )

    if stage == 'stage2':
        return XDDataset(
            args.visual_length,
            args.test_list if is_test else args.train_list,
            is_test,
            label_map,
            use_audio=args.stage2_use_audio,
            audio_root=args.audio_root,
            text_feature_dir=None,
            return_text_feature=False,
            snippet_text_feature_dir=args.snippet_text_feature_dir,
            return_snippet_text_feature=args.stage2_use_snippet_gating,
            use_rag=args.stage2_use_rag,
            rag_topk=args.rag_topk,
            rag_max_bank_size=args.rag_max_bank_size,
            rag_train_list=args.rag_train_list or args.train_list,
        )

    raise ValueError(f'Unsupported dataset stage: {stage}')



def _build_frame_offsets(csv_path):
    full_test_df = pd.read_csv(csv_path)
    frame_counts = full_test_df['path'].map(lambda p: np.load(p).shape[0] * 16).to_numpy(dtype=np.int64)
    return np.concatenate(([0], np.cumsum(frame_counts[:-1], dtype=np.int64)))



def _evaluate_model(model, testloader, args, prompt_text, gt, gtsegments, gtlabels, device, frame_offsets=None, stage='joint'):
    use_dcsa = getattr(args, 'use_dcsa', False)
    temp = getattr(args, 'temp', 1.0)
    test_kwargs = {
        'use_motion_refine': args.use_motion_refine,
        'frame_offsets': frame_offsets,
        'force_visual_only': False,
        'classification_on_pure_visual': False,
        'override_use_debiased_causal_graph': None,
        'use_audio_aux': args.use_audio_aux,
        'use_snippet_text_gating': args.use_snippet_text_gating,
        'use_rag': args.use_rag,
        'use_dcsa': use_dcsa,
        'temp': temp,
        'logits3_alpha': getattr(args, 'logits3_alpha', 0.0),
    }
    if stage == 'stage1':
        test_kwargs.update({
            'force_visual_only': args.stage1_student_force_visual_only,
            'classification_on_pure_visual': args.stage1_student_classification_on_pure_visual,
            'override_use_debiased_causal_graph': args.stage1_student_use_causal_graph,
            'use_audio_aux': False,
            'use_snippet_text_gating': False,
            'use_rag': False,
        })
    elif stage == 'stage2':
        test_kwargs.update({
            'use_audio_aux': args.stage2_use_audio,
            'use_snippet_text_gating': args.stage2_use_snippet_gating,
            'use_rag': args.stage2_use_rag,
            'override_use_debiased_causal_graph': args.stage2_use_causal_graph,
        })
    return test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device, **test_kwargs)



def train_teacher_stage1(model, train_loader, optimizer, prompt_text, label_map, device, args):
    model.to(device)
    model.train()
    loss_total1 = 0.0
    loss_total2 = 0.0

    for i, batch in enumerate(train_loader):
        features, audio_features, labels, lengths, text_features, snippet_text, rag_class_prior, rag_anomaly_score, rag_confidence, _, _ = _unpack_xd_ts_batch(
            batch,
            use_audio_aux=_stage1_teacher_use_audio(args),
            use_text_feature=True,
            use_snippet_text_gating=_stage1_teacher_use_snippet(args),
            use_rag=_stage1_teacher_use_rag(args),
        )
        visual_features = features.to(device)
        feat_lengths = lengths.to(device)
        raw_labels = list(labels)
        text_feature_batch = text_features
        audio_batch = audio_features.to(device) if audio_features is not None else None
        snippet_text_batch = snippet_text.to(device) if snippet_text is not None else None
        rag_features = _build_rag_features(rag_class_prior, rag_anomaly_score, rag_confidence, device, visual_features.dtype)
        text_labels = get_batch_label(raw_labels, prompt_text, label_map).to(device)

        teacher_text = build_teacher_text_features(model, prompt_text, text_feature_batch, device)
        text_features_out, logits1, logits2 = model(
            visual_features, None, prompt_text, feat_lengths,
            text_features_override=teacher_text,
            **_build_teacher_forward_kwargs(args, 'stage1', audio_batch, snippet_text_batch, rag_features, kd_pass=False)
        )

        loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
        loss2 = CLASM(logits2, text_labels, feat_lengths, device)
        loss3 = text_contrast_loss(text_features_out)
        temporal_loss = temporal_consistency_loss(logits1, feat_lengths)
        loss = loss1 + loss2 + loss3 + args.temporal_consistency_weight * temporal_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total1 += loss1.item()
        loss_total2 += loss2.item()

        step = i * train_loader.batch_size
        if step % 1280 == 0 and step != 0:
            print(f'teacher step={step} loss1={loss_total1/(i+1):.4f} loss2={loss_total2/(i+1):.4f}', flush=True)



def train_student_joint(student, teacher, train_loader, testloader, args, label_map, device, frame_offsets=None):
    student.to(device)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    use_dnp = getattr(args, 'use_dnp', False)
    use_dcsa = getattr(args, 'use_dcsa', False)
    loss2_weight = getattr(args, 'loss2_weight', 1.0)
    dcsa_loss_weight = getattr(args, 'dcsa_loss_weight', 1.0)
    dnp_loss_weight = getattr(args, 'dnp_loss_weight', 1.0)
    student_visual_only = getattr(args, 'joint_student_visual_only', False)

    if use_dnp and student.video_anomaly_refiner is not None:
        refiner_params, main_params = [], []
        for name, param in student.named_parameters():
            if not param.requires_grad:
                continue
            if 'video_anomaly_refiner' in name:
                refiner_params.append(param)
            else:
                main_params.append(param)
        optimizer = torch.optim.AdamW(main_params, lr=args.lr)
        total_iters = args.max_epoch * len(train_loader)
        optimizer_refiner = StableAdamW(refiner_params, lr=args.lr,
                                        betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
        scheduler_refiner = WarmCosineScheduler(optimizer_refiner, base_value=args.lr,
                                                final_value=args.lr * 0.1,
                                                total_iters=total_iters, warmup_iters=100)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    else:
        optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
        scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
        optimizer_refiner = None
        scheduler_refiner = None

    prompt_text = get_prompt_text(label_map)
    ap_best = -1.0
    map_best = -1.0
    bestap_ckpt_path = args.checkpoint_path
    bestap_model_path = args.model_path
    bestmap_ckpt_path = _suffix_path(args.checkpoint_path, '_bestMAP')
    bestmap_model_path = _suffix_path(args.model_path, '_bestMAP')

    for e in range(args.max_epoch):
        student.train()
        loss_total1 = 0.0
        loss_total2 = 0.0
        loss_kd_bin = 0.0
        loss_kd_multi = 0.0
        loss_kd_feat = 0.0
        loss_acc = 0.0

        for i, batch in enumerate(train_loader):
            features, audio_features, labels, lengths, text_features, snippet_text, rag_class_prior, rag_anomaly_score, rag_confidence, _, _ = _unpack_xd_ts_batch(
                batch,
                use_audio_aux=args.use_audio_aux,
                use_text_feature=True,
                use_snippet_text_gating=args.use_snippet_text_gating,
                use_rag=args.use_rag,
            )
            visual_features = features.to(device)
            feat_lengths = lengths.to(device)
            raw_labels = list(labels)
            text_feature_batch = text_features
            audio_batch = audio_features.to(device) if audio_features is not None else None
            snippet_text_batch = snippet_text.to(device) if snippet_text is not None else None
            rag_features = _build_rag_features(rag_class_prior, rag_anomaly_score, rag_confidence, device, visual_features.dtype)
            text_labels = get_batch_label(raw_labels, prompt_text, label_map).to(device)

            with torch.no_grad():
                teacher_text = build_teacher_text_features(teacher, prompt_text, text_feature_batch, device)
                teacher_out = teacher(
                    visual_features, None, prompt_text, feat_lengths,
                    text_features_override=teacher_text,
                    return_visual_features=True,
                    return_acc_pseudo_labels=args.acc_dense_distill_weight > 0,
                    acc_eta=args.acc_eta,
                    acc_threshold=args.acc_threshold,
                    **_build_teacher_forward_kwargs(args, 'joint', audio_batch, snippet_text_batch, rag_features)
                )
                t_logits1 = teacher_out[1]
                t_logits2 = teacher_out[2]
                t_visual = teacher_out[3]
                g_teacher = teacher_out[4] if args.acc_dense_distill_weight > 0 else None

            if student_visual_only:
                student_kwargs = {
                    'snippet_text_features': None,
                    'use_motion_refine': args.use_motion_refine,
                    'audio': None,
                    'use_audio_aux': False,
                    'rag_features': None,
                    'force_visual_only': True,
                    'classification_on_pure_visual': True,
                }
            else:
                student_kwargs = _build_student_forward_kwargs(args, 'joint', audio_batch, snippet_text_batch, rag_features)

            fwd_out = student(
                visual_features, None, prompt_text, feat_lengths,
                return_visual_features=True,
                use_dcsa=use_dcsa,
                use_dnp=use_dnp,
                **student_kwargs
            )
            # output order: text, logits1, logits2, [logits3, logits4 if dcsa], [dnp_dict if dnp], visual
            if use_dcsa and use_dnp:
                text_features_out, logits1, logits2, logits3, logits4, dnp_dict, s_visual = fwd_out
            elif use_dcsa:
                text_features_out, logits1, logits2, logits3, logits4, s_visual = fwd_out
                dnp_dict = None
            elif use_dnp:
                text_features_out, logits1, logits2, dnp_dict, s_visual = fwd_out
            else:
                text_features_out, logits1, logits2, s_visual = fwd_out
                dnp_dict = None

            loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss3 = text_contrast_loss(text_features_out)
            temporal_loss = temporal_consistency_loss(logits1, feat_lengths)

            kd_bin = F.mse_loss(torch.sigmoid(logits1), torch.sigmoid(t_logits1).detach())
            T = max(getattr(args, 'kd_temp', 2.0), 1.0)
            kd_multi = F.kl_div(
                F.log_softmax(logits2 / T, dim=-1),
                F.softmax(t_logits2.detach() / T, dim=-1),
                reduction='batchmean'
            ) * (T * T)
            kd_feat = F.mse_loss(F.normalize(s_visual, dim=-1),
                                 F.normalize(t_visual.detach(), dim=-1))
            kd_loss = (
                args.distill_kd_bin_weight * kd_bin
                + args.distill_kd_multi_weight * kd_multi
                + args.distill_kd_feat_weight * kd_feat
            )

            dense_distill_loss = torch.tensor(0.0, device=device)
            if args.acc_dense_distill_weight > 0 and g_teacher is not None:
                q_student = logits2.softmax(dim=-1)
                dense_distill_loss = (
                    q_student - g_teacher.to(device=q_student.device, dtype=q_student.dtype)
                ).abs().mean()

            loss = (
                args.loss1_weight * loss1
                + loss2_weight * loss2
                + loss3
                + args.distill_weight * kd_loss
                + args.temporal_consistency_weight * temporal_loss
                + args.acc_dense_distill_weight * dense_distill_loss
            )

            if use_dcsa:
                loss4 = CLASM_EVENT(logits3, text_labels, feat_lengths, device)
                loss5 = CLASM_BKG(logits4, text_labels, feat_lengths, device)
                loss = loss + dcsa_loss_weight * (loss4 + loss5)

            if use_dnp and dnp_dict is not None:
                consistency_loss = _consistency_loss_fn(
                    logits1, dnp_dict['original_features'], dnp_dict['reconstructed_features'], feat_lengths
                )
                loss = loss + dnp_loss_weight * (consistency_loss + dnp_dict['g_loss'])

            optimizer.zero_grad()
            if optimizer_refiner is not None:
                optimizer_refiner.zero_grad()
            loss.backward()
            optimizer.step()
            if optimizer_refiner is not None:
                optimizer_refiner.step()
                scheduler_refiner.step()

            loss_total1 += loss1.item()
            loss_total2 += loss2.item()
            loss_kd_bin += kd_bin.item()
            loss_kd_multi += kd_multi.item()
            loss_kd_feat += kd_feat.item()
            loss_acc += dense_distill_loss.item()

            step = i * train_loader.batch_size
            if step % 1280 == 0 and step != 0:
                print(
                    f'epoch={e+1} step={step} loss1={loss_total1/(i+1):.4f} '
                    f'loss2={loss_total2/(i+1):.4f} '
                    f'kd_bin={loss_kd_bin/(i+1):.4f} kd_multi={loss_kd_multi/(i+1):.4f} '
                    f'kd_feat={loss_kd_feat/(i+1):.4f} acc={loss_acc/(i+1):.4f}',
                    flush=True,
                )

        scheduler.step()
        print(f'epoch={e+1} final evaluation', flush=True)
        AUC, AP, avg_map = _evaluate_model(
            student, testloader, args, prompt_text, gt, gtsegments, gtlabels, device,
            frame_offsets=frame_offsets, stage='joint'
        )
        if AP > ap_best:
            ap_best = float(AP)
            checkpoint = {
                'epoch': e,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ap': ap_best,
                'auc1': float(AUC),
                'map': float(avg_map),
            }
            torch.save(checkpoint, bestap_ckpt_path)
            torch.save(student.state_dict(), bestap_model_path)
            print(f'Saved bestAP ckpt epoch={e+1} AP={ap_best:.6f} MAP={avg_map:.2f}', flush=True)
        if avg_map > map_best:
            map_best = float(avg_map)
            checkpoint = {
                'epoch': e,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ap': float(AP),
                'auc1': float(AUC),
                'map': map_best,
            }
            torch.save(checkpoint, bestmap_ckpt_path)
            torch.save(student.state_dict(), bestmap_model_path)
            print(f'Saved bestMAP ckpt epoch={e+1} AP={float(AP):.6f} MAP={map_best:.2f}', flush=True)

    print(f'Finished. bestAP={ap_best:.6f} bestMAP={map_best:.2f}', flush=True)
    restore_target = bestmap_ckpt_path if os.path.isfile(bestmap_ckpt_path) else bestap_ckpt_path
    print(f'Restoring final weights from {restore_target}', flush=True)
    _try_restore_best(student, restore_target)



def train_student_stage1(student, teacher, train_loader, testloader, args, label_map, device, frame_offsets=None):
    student.to(device)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    use_dnp = getattr(args, 'use_dnp', False)
    use_dcsa = getattr(args, 'use_dcsa', False)
    loss2_weight = getattr(args, 'loss2_weight', 1.0)

    if use_dnp and student.video_anomaly_refiner is not None:
        refiner_params, main_params = [], []
        for name, param in student.named_parameters():
            if not param.requires_grad:
                continue
            if 'video_anomaly_refiner' in name:
                refiner_params.append(param)
            else:
                main_params.append(param)
        optimizer = torch.optim.AdamW(main_params, lr=args.lr)
        total_iters = args.max_epoch * len(train_loader)
        optimizer_refiner = StableAdamW(refiner_params, lr=args.lr,
                                        betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
        scheduler_refiner = WarmCosineScheduler(optimizer_refiner, base_value=args.lr,
                                                final_value=args.lr * 0.1,
                                                total_iters=total_iters, warmup_iters=100)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    else:
        optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
        scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
        optimizer_refiner = None
        scheduler_refiner = None

    prompt_text = get_prompt_text(label_map)
    ap_best = -1.0

    for e in range(args.max_epoch):
        student.train()
        loss_total1 = 0.0
        loss_total2 = 0.0
        loss_kd = 0.0

        for i, batch in enumerate(train_loader):
            features, audio_features, labels, lengths, text_features, snippet_text, rag_class_prior, rag_anomaly_score, rag_confidence, _, _ = _unpack_xd_ts_batch(
                batch,
                use_audio_aux=_stage1_teacher_use_audio(args),
                use_text_feature=True,
                use_snippet_text_gating=_stage1_teacher_use_snippet(args),
                use_rag=_stage1_teacher_use_rag(args),
            )
            visual_features = features.to(device)
            feat_lengths = lengths.to(device)
            raw_labels = list(labels)
            text_feature_batch = text_features
            audio_batch = audio_features.to(device) if audio_features is not None else None
            snippet_text_batch = snippet_text.to(device) if snippet_text is not None else None
            rag_features = _build_rag_features(rag_class_prior, rag_anomaly_score, rag_confidence, device, visual_features.dtype)
            text_labels = get_batch_label(raw_labels, prompt_text, label_map).to(device)

            with torch.no_grad():
                teacher_text = build_teacher_text_features(teacher, prompt_text, text_feature_batch, device)
                teacher_out = teacher(
                    visual_features, None, prompt_text, feat_lengths,
                    text_features_override=teacher_text,
                    return_visual_features=True,
                    **_build_teacher_forward_kwargs(args, 'stage1', audio_batch, snippet_text_batch, rag_features, kd_pass=True)
                )
                t_logits1 = teacher_out[1]
                t_logits2 = teacher_out[2]
                t_visual = teacher_out[3]

            fwd_out = student(
                visual_features, None, prompt_text, feat_lengths,
                return_visual_features=True,
                use_dcsa=use_dcsa,
                use_dnp=use_dnp,
                **_build_student_forward_kwargs(args, 'stage1')
            )
            # output order: text, logits1, logits2, [logits3, logits4 if dcsa], [dnp_dict if dnp], visual
            if use_dcsa and use_dnp:
                text_features_out, logits1, logits2, logits3, logits4, dnp_dict, s_visual = fwd_out
            elif use_dcsa:
                text_features_out, logits1, logits2, logits3, logits4, s_visual = fwd_out
                dnp_dict = None
            elif use_dnp:
                text_features_out, logits1, logits2, dnp_dict, s_visual = fwd_out
            else:
                text_features_out, logits1, logits2, s_visual = fwd_out
                dnp_dict = None

            loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss3 = text_contrast_loss(text_features_out)
            temporal_loss = temporal_consistency_loss(logits1, feat_lengths)
            event_shape_loss, event_trend_loss = event_completeness_distillation(logits1, t_logits1, feat_lengths)
            kd_bin = F.mse_loss(torch.sigmoid(logits1), torch.sigmoid(t_logits1))
            kd_multi = F.kl_div(
                F.log_softmax(logits2 / 2.0, dim=-1),
                F.softmax(t_logits2 / 2.0, dim=-1),
                reduction='sum'
            ) * 4.0 / (logits2.shape[0] * logits2.shape[1])
            kd_feat = F.mse_loss(F.normalize(s_visual, dim=-1), F.normalize(t_visual, dim=-1))
            kd_loss = (
                args.distill_kd_bin_weight * kd_bin
                + args.distill_kd_multi_weight * kd_multi
                + args.distill_kd_feat_weight * kd_feat
            )

            loss = (
                args.loss1_weight * loss1
                + loss2_weight * loss2
                + loss3
                + args.distill_weight * kd_loss
                + args.temporal_consistency_weight * temporal_loss
                + args.event_kd_weight * event_shape_loss
                + args.event_trend_weight * event_trend_loss
            )

            if use_dcsa:
                loss4 = CLASM_EVENT(logits3, text_labels, feat_lengths, device)
                loss5 = CLASM_BKG(logits4, text_labels, feat_lengths, device)
                loss = loss + loss4 + loss5

            if use_dnp and dnp_dict is not None:
                consistency_loss = _consistency_loss_fn(
                    logits1, dnp_dict['original_features'], dnp_dict['reconstructed_features'], feat_lengths
                )
                loss = loss + consistency_loss + dnp_dict['g_loss']

            optimizer.zero_grad()
            if optimizer_refiner is not None:
                optimizer_refiner.zero_grad()
            loss.backward()
            optimizer.step()
            if optimizer_refiner is not None:
                optimizer_refiner.step()
                scheduler_refiner.step()

            loss_total1 += loss1.item()
            loss_total2 += loss2.item()
            loss_kd += kd_loss.item()

            step = i * train_loader.batch_size
            if step % 1280 == 0 and step != 0:
                print(
                    f'stage1 epoch={e+1} step={step} loss1={loss_total1/(i+1):.4f} '
                    f'loss2={loss_total2/(i+1):.4f} kd={loss_kd/(i+1):.4f}',
                    flush=True,
                )

        scheduler.step()
        print(f'stage1 epoch={e+1} final evaluation', flush=True)
        AUC, AP, avg_map = _evaluate_model(
            student, testloader, args, prompt_text, gt, gtsegments, gtlabels, device,
            frame_offsets=frame_offsets, stage='stage1'
        )
        if AP > ap_best:
            ap_best = float(AP)
            checkpoint = {
                'epoch': e,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ap': ap_best,
                'auc1': float(AUC),
                'map': float(avg_map),
            }
            torch.save(checkpoint, args.checkpoint_path)
            torch.save(student.state_dict(), args.model_path)

    _try_restore_best(student, args.checkpoint_path)



def train_student_stage2(student, train_loader, testloader, args, label_map, device, frame_offsets=None):
    student.to(device)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    use_dnp = getattr(args, 'use_dnp', False)
    use_dcsa = getattr(args, 'use_dcsa', False)
    loss2_weight = getattr(args, 'loss2_weight', 1.0)

    if use_dnp and student.video_anomaly_refiner is not None:
        refiner_params, main_params = [], []
        for name, param in student.named_parameters():
            if not param.requires_grad:
                continue
            if 'video_anomaly_refiner' in name:
                refiner_params.append(param)
            else:
                main_params.append(param)
        optimizer = torch.optim.AdamW(main_params, lr=args.lr)
        total_iters = args.max_epoch * len(train_loader)
        optimizer_refiner = StableAdamW(refiner_params, lr=args.lr,
                                        betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
        scheduler_refiner = WarmCosineScheduler(optimizer_refiner, base_value=args.lr,
                                                final_value=args.lr * 0.1,
                                                total_iters=total_iters, warmup_iters=100)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    else:
        optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
        scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
        optimizer_refiner = None
        scheduler_refiner = None

    prompt_text = get_prompt_text(label_map)
    ap_best = -1.0

    for e in range(args.max_epoch):
        student.train()
        loss_total1 = 0.0
        loss_total2 = 0.0

        for i, batch in enumerate(train_loader):
            features, audio_features, labels, lengths, _, snippet_text, rag_class_prior, rag_anomaly_score, rag_confidence, _, _ = _unpack_xd_ts_batch(
                batch,
                use_audio_aux=args.stage2_use_audio,
                use_text_feature=False,
                use_snippet_text_gating=args.stage2_use_snippet_gating,
                use_rag=args.stage2_use_rag,
            )
            visual_features = features.to(device)
            feat_lengths = lengths.to(device)
            raw_labels = list(labels)
            audio_batch = audio_features.to(device) if audio_features is not None else None
            snippet_text_batch = snippet_text.to(device) if snippet_text is not None else None
            rag_features = _build_rag_features(rag_class_prior, rag_anomaly_score, rag_confidence, device, visual_features.dtype)
            text_labels = get_batch_label(raw_labels, prompt_text, label_map).to(device)

            fwd_out = student(
                visual_features, None, prompt_text, feat_lengths,
                use_dcsa=use_dcsa,
                use_dnp=use_dnp,
                **_build_student_forward_kwargs(args, 'stage2', audio_batch, snippet_text_batch, rag_features)
            )
            if use_dcsa and use_dnp:
                text_features_out, logits1, logits2, logits3, logits4, dnp_dict = fwd_out
            elif use_dcsa:
                text_features_out, logits1, logits2, logits3, logits4 = fwd_out
                dnp_dict = None
            elif use_dnp:
                text_features_out, logits1, logits2, dnp_dict = fwd_out
            else:
                text_features_out, logits1, logits2 = fwd_out
                dnp_dict = None

            loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss3 = text_contrast_loss(text_features_out)
            temporal_loss = temporal_consistency_loss(logits1, feat_lengths)

            loss = (
                args.loss1_weight * loss1
                + loss2_weight * loss2
                + loss3
                + args.temporal_consistency_weight * temporal_loss
            )

            if use_dcsa:
                loss4 = CLASM_EVENT(logits3, text_labels, feat_lengths, device)
                loss5 = CLASM_BKG(logits4, text_labels, feat_lengths, device)
                loss = loss + loss4 + loss5

            if use_dnp and dnp_dict is not None:
                consistency_loss = _consistency_loss_fn(
                    logits1, dnp_dict['original_features'], dnp_dict['reconstructed_features'], feat_lengths
                )
                loss = loss + consistency_loss + dnp_dict['g_loss']

            optimizer.zero_grad()
            if optimizer_refiner is not None:
                optimizer_refiner.zero_grad()
            loss.backward()
            optimizer.step()
            if optimizer_refiner is not None:
                optimizer_refiner.step()
                scheduler_refiner.step()

            loss_total1 += loss1.item()
            loss_total2 += loss2.item()

            step = i * train_loader.batch_size
            if step % 1280 == 0 and step != 0:
                print(
                    f'stage2 epoch={e+1} step={step} loss1={loss_total1/(i+1):.4f} '
                    f'loss2={loss_total2/(i+1):.4f}',
                    flush=True,
                )

        scheduler.step()
        print(f'stage2 epoch={e+1} final evaluation', flush=True)
        AUC, AP, avg_map = _evaluate_model(
            student, testloader, args, prompt_text, gt, gtsegments, gtlabels, device,
            frame_offsets=frame_offsets, stage='stage2'
        )
        if AP > ap_best:
            ap_best = float(AP)
            checkpoint = {
                'epoch': e,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ap': ap_best,
                'auc1': float(AUC),
                'map': float(avg_map),
            }
            torch.save(checkpoint, args.stage2_checkpoint_path)
            torch.save(student.state_dict(), args.stage2_model_path)

    _try_restore_best(student, args.stage2_checkpoint_path)



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = xd_option.parser.parse_args()
    args = xd_option.resolve_text_feature_dirs(args)
    setup_seed(args.seed)
    active_log_path = args.stage2_log_path if args.train_stage == 'stage2' else args.log_path
    log_file = _setup_logging(active_log_path)

    label_map = {
        'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot',
        'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'
    }

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    os.makedirs(os.path.dirname(args.model_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.teacher_model_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.stage2_model_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.stage2_checkpoint_path) or '.', exist_ok=True)

    teacher = CLIPVAD(
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
        use_clip_adapter=getattr(args, 'use_clip_adapter', False),
        clip_adapter_layers=getattr(args, 'clip_adapter_layers', 3),
        clip_adapter_weight=getattr(args, 'clip_adapter_weight', 0.1),
    )

    teacher_path = args.teacher_model_path
    teacher_best_path = _teacher_best_path(teacher_path)

    if args.train_stage in {'joint', 'stage1', 'two_stage'}:
        teacher_train_dataset = _build_dataset('stage1' if args.train_stage in {'stage1', 'two_stage'} else 'joint', args, label_map, is_test=False)
        teacher_test_dataset = _build_dataset('stage1' if args.train_stage in {'stage1', 'two_stage'} else 'joint', args, label_map, is_test=True)
        teacher_train_loader = DataLoader(teacher_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        teacher_test_loader = DataLoader(teacher_test_dataset, batch_size=1, shuffle=False)
        teacher_frame_offsets = _build_frame_offsets(args.test_list) if (_stage1_teacher_use_audio(args) if args.train_stage in {'stage1', 'two_stage'} else args.use_audio_aux) else None

        if args.train_teacher:
            _load_pretrained_if_available(teacher, args.pretrained_path)
            teacher_optimizer = torch.optim.AdamW(teacher.parameters(), lr=args.teacher_lr)
            teacher_scheduler = MultiStepLR(teacher_optimizer, args.teacher_scheduler_milestones, args.scheduler_rate)
            ap_best = -1.0
            for te in range(args.teacher_epochs):
                if args.train_stage in {'stage1', 'two_stage'}:
                    train_teacher_stage1(teacher, teacher_train_loader, teacher_optimizer, prompt_text, label_map, device, args)
                else:
                    train_teacher_stage1(teacher, teacher_train_loader, teacher_optimizer, prompt_text, label_map, device, args)
                teacher_scheduler.step()
                print(f'teacher epoch {te+1}/{args.teacher_epochs} done', flush=True)
                teacher_eval_stage = 'stage1' if args.train_stage in {'stage1', 'two_stage'} else 'joint'
                teacher_auc, teacher_ap, teacher_map = _evaluate_model(
                    teacher, teacher_test_loader, args, prompt_text, gt, gtsegments, gtlabels, device,
                    frame_offsets=teacher_frame_offsets, stage=teacher_eval_stage
                )
                if teacher_ap > ap_best:
                    ap_best = float(teacher_ap)
                    checkpoint = {
                        'epoch': te,
                        'model_state_dict': teacher.state_dict(),
                        'optimizer_state_dict': teacher_optimizer.state_dict(),
                        'ap': float(teacher_ap),
                        'auc1': float(teacher_auc),
                        'map': float(teacher_map),
                    }
                    torch.save(checkpoint, teacher_best_path)
                    print(f'Saved best teacher to {teacher_best_path} (AP={teacher_ap:.6f})', flush=True)
            if os.path.isfile(teacher_best_path):
                _try_restore_best(teacher, teacher_best_path)
            torch.save(teacher.state_dict(), teacher_path)
        else:
            load_teacher_path = teacher_best_path if os.path.isfile(teacher_best_path) else teacher_path
            if not os.path.isfile(load_teacher_path):
                raise FileNotFoundError(f'Teacher checkpoint not found at {load_teacher_path}. Re-run with --train-teacher to create it.')
            print(f'Loading existing teacher from {load_teacher_path}...', flush=True)
            teacher_payload = _safe_torch_load(load_teacher_path, map_location='cpu')
            teacher_state = _extract_state_dict(teacher_payload)
            if teacher_state is None:
                raise ValueError(f'Unsupported teacher format: {load_teacher_path}')
            missing, unexpected = teacher.load_state_dict(teacher_state, strict=False)
            print(f'Loaded teacher with strict=False. Missing: {len(missing)}, Unexpected: {len(unexpected)}', flush=True)

    student = CLIPVAD(
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

    if args.train_stage == 'stage2' and args.stage2_init_model_path:
        _load_pretrained_if_available(student, args.stage2_init_model_path)
    else:
        _load_pretrained_if_available(student, args.pretrained_path)

    if args.train_stage == 'joint':
        train_dataset = _build_dataset('joint', args, label_map, is_test=False)
        test_dataset = _build_dataset('joint', args, label_map, is_test=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        frame_offsets = _build_frame_offsets(args.test_list) if args.use_audio_aux else None
        train_student_joint(student, teacher, train_loader, test_loader, args, label_map, device, frame_offsets=frame_offsets)
    elif args.train_stage == 'stage1':
        train_dataset = _build_dataset('stage1', args, label_map, is_test=False)
        test_dataset = _build_dataset('stage1', args, label_map, is_test=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        frame_offsets = _build_frame_offsets(args.test_list) if _stage1_teacher_use_audio(args) else None
        train_student_stage1(student, teacher, train_loader, test_loader, args, label_map, device, frame_offsets=frame_offsets)
    elif args.train_stage == 'stage2':
        train_dataset = _build_dataset('stage2', args, label_map, is_test=False)
        test_dataset = _build_dataset('stage2', args, label_map, is_test=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        frame_offsets = _build_frame_offsets(args.test_list) if args.stage2_use_audio else None
        train_student_stage2(student, train_loader, test_loader, args, label_map, device, frame_offsets=frame_offsets)
    elif args.train_stage == 'two_stage':
        stage1_train_dataset = _build_dataset('stage1', args, label_map, is_test=False)
        stage1_test_dataset = _build_dataset('stage1', args, label_map, is_test=True)
        stage1_train_loader = DataLoader(stage1_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        stage1_test_loader = DataLoader(stage1_test_dataset, batch_size=1, shuffle=False)
        stage1_frame_offsets = _build_frame_offsets(args.test_list) if _stage1_teacher_use_audio(args) else None
        train_student_stage1(student, teacher, stage1_train_loader, stage1_test_loader, args, label_map, device, frame_offsets=stage1_frame_offsets)
        if args.stage2_init_model_path and os.path.isfile(args.stage2_init_model_path):
            _load_pretrained_if_available(student, args.stage2_init_model_path)
        else:
            _try_restore_best(student, args.checkpoint_path)
        stage2_train_dataset = _build_dataset('stage2', args, label_map, is_test=False)
        stage2_test_dataset = _build_dataset('stage2', args, label_map, is_test=True)
        stage2_train_loader = DataLoader(stage2_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        stage2_test_loader = DataLoader(stage2_test_dataset, batch_size=1, shuffle=False)
        stage2_frame_offsets = _build_frame_offsets(args.test_list) if args.stage2_use_audio else None
        train_student_stage2(student, stage2_train_loader, stage2_test_loader, args, label_map, device, frame_offsets=stage2_frame_offsets)
    else:
        raise ValueError(f'Unsupported train stage: {args.train_stage}')

    log_file.close()
