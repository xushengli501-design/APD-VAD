import os
import contextlib
import sys
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random

from model_ucf import CLIPVAD
from ucf_test import test
from utils.dataset import UCFDataset, _canonical_video_id
from utils.tools import get_prompt_text, get_batch_label
from utils.StableAdamW import StableAdamW
import ucf_option


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


def _load_pretrained_if_available(model, pretrained_path: str):
    if not pretrained_path:
        return
    if not os.path.exists(pretrained_path):
        print(f"Pretrained not found: {pretrained_path}")
        return
    print(f"Loading pretrained: {pretrained_path}")
    state_dict = _safe_torch_load(pretrained_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded with strict=False. Missing: {len(missing)}, Unexpected: {len(unexpected)}")


def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    return -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)


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
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)
    labels2 = torch.full(labels.shape, 0.01, device=labels.device)
    labels2[:, 0] = 1
    labels2_sum = labels2.sum(dim=1, keepdim=True).clamp(min=1e-6)
    labels2 = (1 - epsilon) * (labels2 / labels2_sum) + epsilon / num_classes
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=1, largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    return -torch.mean(torch.sum(labels2 * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)


class ConsistencyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, logits1, original_features, reconstructed_features, lengths):
        recon_error = 1.0 - F.cosine_similarity(original_features, reconstructed_features, dim=-1)
        recon_error = recon_error / 2.0
        classifier_prob = torch.sigmoid(logits1.squeeze(-1))
        B, N = logits1.shape[0], logits1.shape[1]
        mask = torch.arange(N, device=logits1.device)[None, :] < lengths[:, None]
        return self.mse_loss(classifier_prob[mask], recon_error[mask])


consistency_loss_fn = ConsistencyLoss()


class WarmCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
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


def CLAS2(logits, labels, lengths, device, topk_scores=None, use_adaptive_mil=False, adaptive_temp=0.1):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device=device, dtype=logits.dtype)
    probs = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0).clamp_(1e-6, 1 - 1e-6)
    if topk_scores is None:
        topk_scores = probs
    else:
        topk_scores = topk_scores.reshape(logits.shape[0], logits.shape[1])
        topk_scores = torch.nan_to_num(topk_scores, nan=0.0, posinf=1.0, neginf=0.0)
    for i in range(logits.shape[0]):
        valid_length = int(lengths[i].item())
        if use_adaptive_mil:
            ref = topk_scores[i, 0:valid_length]
            attn = torch.softmax(ref / max(adaptive_temp, 1e-6), dim=0)
            tmp = torch.sum(attn * probs[i, 0:valid_length]).view(1)
        else:
            k = int(lengths[i] / 16 + 1)
            _, indexes = torch.topk(topk_scores[i, 0:valid_length], k=k, largest=True)
            tmp = torch.mean(probs[i, indexes]).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)
    instance_logits = torch.nan_to_num(instance_logits, nan=0.5, posinf=1.0, neginf=0.0).clamp_(1e-6, 1 - 1e-6)
    return F.binary_cross_entropy(instance_logits, labels)


def compute_text_topk_scores(visual_features, snippet_text_features, temperature=0.1):
    visual_norm = F.normalize(visual_features, dim=-1)
    text_norm = F.normalize(snippet_text_features.to(visual_features.device, dtype=visual_features.dtype), dim=-1)
    return torch.sigmoid((visual_norm * text_norm).sum(dim=-1) / max(temperature, 1e-6))


def text_contrast_loss(text_features):
    if text_features.dim() == 3:
        text_features = text_features.mean(dim=0)
    loss = torch.zeros(1, device=text_features.device)
    text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
    for j in range(1, text_features.shape[0]):
        text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
        loss += torch.abs(text_feature_normal @ text_feature_abr)
    return loss / 13 * 1e-1


def hard_negative_margin_loss(logits, lengths, text_labels, topk=3, margin=0.5):
    probs = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    video_scores = []
    video_is_abnormal = []
    for i in range(probs.shape[0]):
        valid_length = int(lengths[i].item())
        if valid_length <= 0:
            continue
        k = min(max(int(topk), 1), valid_length)
        topk_vals = torch.topk(probs[i, :valid_length], k=k, largest=True).values
        video_scores.append(topk_vals.mean())
        video_is_abnormal.append((1.0 - text_labels[i, 0]).detach() > 0.5)

    if not video_scores:
        return probs.new_tensor(0.0)

    video_scores = torch.stack(video_scores, dim=0)
    video_is_abnormal = torch.tensor(video_is_abnormal, device=video_scores.device, dtype=torch.bool)
    normal_scores = video_scores[~video_is_abnormal]
    abnormal_scores = video_scores[video_is_abnormal]

    if normal_scores.numel() == 0 or abnormal_scores.numel() == 0:
        return probs.new_tensor(0.0)

    pairwise = normal_scores.unsqueeze(0) - abnormal_scores.unsqueeze(1) + margin
    return torch.relu(pairwise).mean()


def dcsa_loss(visual_features, logits1, text_features, text_labels, lengths, temperature=0.07):
    """Decoupled Contrastive Semantic Alignment (DSANet DCSA).

    Uses detached logits1 so DCSA gradients don't affect the binary classifier.
    """
    s_det = torch.sigmoid(logits1.squeeze(-1).detach())  # [B, T], stop-gradient

    w_event = torch.zeros_like(s_det)
    w_bkg = torch.zeros_like(s_det)
    for i in range(s_det.shape[0]):
        valid_len = int(lengths[i].item())
        if valid_len <= 0:
            continue
        w_e = torch.softmax(s_det[i, :valid_len], dim=0)
        w_b = (1.0 - w_e).clamp_min(0.0)
        w_b = w_b / w_b.sum().clamp_min(1e-6)
        w_event[i, :valid_len] = w_e
        w_bkg[i, :valid_len] = w_b

    f_event = (w_event.unsqueeze(-1) * visual_features).sum(dim=1)  # [B, D]
    f_bkg = (w_bkg.unsqueeze(-1) * visual_features).sum(dim=1)      # [B, D]
    f_event = F.normalize(f_event, dim=-1)
    f_bkg = F.normalize(f_bkg, dim=-1)

    if text_features.dim() == 3:
        text_features = text_features.mean(dim=0)
    text_norm = F.normalize(text_features.detach().to(dtype=visual_features.dtype), dim=-1)  # [C, D]

    temp = max(temperature, 1e-6)
    sim_event = f_event @ text_norm.t() / temp  # [B, C]
    sim_bkg = f_bkg @ text_norm.t() / temp      # [B, C]

    targets_event = torch.argmax(text_labels.to(sim_event.device), dim=1)  # [B]
    targets_bkg = torch.zeros(f_bkg.shape[0], dtype=torch.long, device=f_bkg.device)

    l_event = F.cross_entropy(sim_event, targets_event)
    l_bkg = F.cross_entropy(sim_bkg, targets_bkg)
    return l_event + l_bkg


def sgnm_consistency_loss(logits1, s_rec, lengths):
    """L_consist = MSE(S_det, S_rec) over valid frames (DSANet Eq. 4)."""
    s_det = torch.sigmoid(logits1.squeeze(-1))  # [B, T]
    total = s_det.new_tensor(0.0)
    count = 0
    for i in range(s_det.shape[0]):
        valid_len = int(lengths[i].item())
        if valid_len <= 0:
            continue
        total = total + F.mse_loss(s_det[i, :valid_len], s_rec[i, :valid_len].detach())
        count += 1
    return total / max(count, 1)


def text_separation_loss(text_features):
    """L_sep: push Normal text away from all anomaly class embeddings (DSANet Eq. 6)."""
    if text_features.dim() == 3:
        text_features = text_features.mean(dim=0)
    t0 = F.normalize(text_features[0], dim=-1)
    loss = text_features.new_tensor(0.0)
    for a in range(1, text_features.shape[0]):
        ta = F.normalize(text_features[a], dim=-1)
        loss = loss + torch.abs(t0 @ ta)
    return loss / max(text_features.shape[0] - 1, 1)


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


def build_teacher_text_features(model, prompt_text, text_feature_batch, device, retrieved_text_batch=None, retrieval_weight=0.0):
    class_text = model.encode_textprompt(prompt_text)
    class_text = F.normalize(torch.nan_to_num(class_text, nan=0.0, posinf=0.0, neginf=0.0), dim=-1)
    caption_text = text_feature_batch.to(device=device, dtype=class_text.dtype)
    caption_text = F.normalize(torch.nan_to_num(caption_text, nan=0.0, posinf=0.0, neginf=0.0), dim=-1)
    if retrieved_text_batch is not None and retrieval_weight > 0:
        retrieved_text = retrieved_text_batch.to(device=device, dtype=class_text.dtype)
        retrieved_text = F.normalize(torch.nan_to_num(retrieved_text, nan=0.0, posinf=0.0, neginf=0.0), dim=-1)
        caption_text = F.normalize((1 - retrieval_weight) * caption_text + retrieval_weight * retrieved_text, dim=-1)
    normal_anchor = class_text[0].unsqueeze(0)
    teacher_text = class_text.unsqueeze(0) + caption_text.unsqueeze(1) * normal_anchor.unsqueeze(1)
    teacher_text = F.normalize(torch.nan_to_num(teacher_text, nan=0.0, posinf=0.0, neginf=0.0), dim=-1)
    return teacher_text


def pool_valid_visual_features(features, lengths):
    pooled = torch.zeros(features.shape[0], features.shape[-1], device=features.device, dtype=features.dtype)
    for i, length in enumerate(lengths):
        valid_length = int(length.item()) if torch.is_tensor(length) else int(length)
        if valid_length <= 0:
            continue
        pooled[i] = features[i, :valid_length].mean(dim=0)
    pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(pooled, dim=-1)


def build_train_retrieval_bank(normal_dataset, anomaly_dataset, args):
    if not args.use_train_visual_retrieval:
        return None

    visual_bank = []
    text_bank = []
    video_ids = []
    seen = set()

    for dataset in (normal_dataset, anomaly_dataset):
        for _, row in dataset.df.iterrows():
            clip_path = row['path']
            canonical_id = _canonical_video_id(clip_path)
            if canonical_id in seen:
                continue
            text_feature = dataset._get_text_feature(clip_path) if hasattr(dataset, '_get_text_feature') else None
            if text_feature is None:
                continue
            text_feature = torch.nan_to_num(text_feature.float(), nan=0.0, posinf=0.0, neginf=0.0)
            if torch.count_nonzero(text_feature).item() == 0:
                continue
            visual_feature = np.load(clip_path).astype(np.float32)
            if visual_feature.ndim != 2 or visual_feature.shape[0] == 0:
                continue
            pooled_visual = torch.from_numpy(visual_feature.mean(axis=0)).float()
            pooled_visual = torch.nan_to_num(pooled_visual, nan=0.0, posinf=0.0, neginf=0.0)
            if torch.count_nonzero(pooled_visual).item() == 0:
                continue
            visual_bank.append(F.normalize(pooled_visual, dim=0))
            text_bank.append(F.normalize(text_feature, dim=0))
            video_ids.append(canonical_id)
            seen.add(canonical_id)

    if not visual_bank:
        return None

    return {
        'video_ids': video_ids,
        'visual_bank': torch.stack(visual_bank, dim=0),
        'text_bank': torch.stack(text_bank, dim=0),
    }


def retrieve_neighbor_text_features(query_visual_batch, query_ids, lengths, retrieval_bank, args, device):
    if retrieval_bank is None or query_ids is None:
        return None

    query_visual = pool_valid_visual_features(query_visual_batch.to(device), lengths)
    visual_bank = retrieval_bank['visual_bank'].to(device=device, dtype=query_visual.dtype)
    text_bank = retrieval_bank['text_bank'].to(device=device, dtype=query_visual.dtype)
    similarities = torch.nan_to_num(query_visual @ visual_bank.t(), nan=-1e4, posinf=1.0, neginf=-1.0)

    for row, query_id in enumerate(query_ids):
        if query_id is None:
            continue
        matches = [idx for idx, bank_id in enumerate(retrieval_bank['video_ids']) if bank_id == query_id]
        if matches:
            similarities[row, matches[0]] = -1e4

    topk = min(args.retrieval_topk, visual_bank.shape[0])
    if topk <= 0:
        return None
    values, indices = torch.topk(similarities, k=topk, dim=-1)
    weights = torch.softmax(values / max(args.retrieval_temp, 1e-6), dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0)
    gathered = text_bank[indices]
    retrieved = torch.sum(gathered * weights.unsqueeze(-1), dim=1)
    retrieved = torch.nan_to_num(retrieved, nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(retrieved, dim=-1)
def _try_restore_best(model, checkpoint_path: str):
    if not os.path.isfile(checkpoint_path):
        return False
    checkpoint = _safe_torch_load(checkpoint_path)
    state_dict = checkpoint.get('model_state_dict') if isinstance(checkpoint, dict) else None
    if state_dict is None:
        return False
    try:
        model.load_state_dict(state_dict)
        return True
    except RuntimeError as exc:
        print(f"Skip incompatible checkpoint {checkpoint_path}: {exc}", flush=True)
        return False


def _teacher_best_path(teacher_model_path: str) -> str:
    if teacher_model_path.endswith('.pth'):
        return teacher_model_path[:-4] + '_best.pth'
    return teacher_model_path + '_best'


def _setup_logging(log_path: str):
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    log_file = open(log_path, 'a', buffering=1)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    print(f"\n===== UCF Teacher-Student training started at {datetime.now().isoformat(timespec='seconds')} =====", flush=True)
    print(f"Logging to {log_path}", flush=True)
    return log_file


def _unpack_batch(batch, use_snippet_text_gating=False, use_train_visual_retrieval=False):
    metadata = None
    if use_snippet_text_gating and use_train_visual_retrieval:
        features, labels, lengths, text_features, snippet_text_features, metadata = batch
        return features, labels, lengths, text_features, snippet_text_features, metadata
    if use_snippet_text_gating:
        features, labels, lengths, text_features, snippet_text_features = batch
        return features, labels, lengths, text_features, snippet_text_features, metadata
    if use_train_visual_retrieval:
        features, labels, lengths, text_features, metadata = batch
        return features, labels, lengths, text_features, None, metadata
    features, labels, lengths, text_features = batch
    return features, labels, lengths, text_features, None, metadata


def _retrieval_kwargs(args, retrieved_visual_batch=None):
    use_retrieved_visual = (
        args.use_train_visual_retrieval
        and getattr(args, 'retrieval_target', 'visual') == 'visual'
        and retrieved_visual_batch is not None
    )
    return {
        'retrieved_visual_features': retrieved_visual_batch,
        'use_retrieved_visual': use_retrieved_visual,
        'retrieval_fuse': getattr(args, 'retrieval_fuse', 'add'),
        'retrieval_weight': getattr(args, 'retrieval_weight', 0.0),
    }


def run_teacher_epoch(model, normal_loader, anomaly_loader, optimizer, prompt_text, label_map, device,
                      use_snippet_text_gating=False, retrieval_bank=None, args=None):
    model.train()
    loss_total1 = loss_total2 = 0.0
    normal_iter = iter(normal_loader)
    anomaly_iter = iter(anomaly_loader)

    for i in range(min(len(normal_loader), len(anomaly_loader))):
        normal_features, normal_label, normal_lengths, normal_text_features, normal_snippet_text, normal_meta = _unpack_batch(
            next(normal_iter), use_snippet_text_gating, args.use_train_visual_retrieval if args is not None else False
        )
        anomaly_features, anomaly_label, anomaly_lengths, anomaly_text_features, anomaly_snippet_text, anomaly_meta = _unpack_batch(
            next(anomaly_iter), use_snippet_text_gating, args.use_train_visual_retrieval if args is not None else False
        )

        visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
        feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
        raw_labels = list(normal_label) + list(anomaly_label)
        snippet_text_batch = None
        if use_snippet_text_gating:
            snippet_text_batch = torch.cat([normal_snippet_text, anomaly_snippet_text], dim=0).to(device)
        text_labels = get_batch_label(raw_labels, prompt_text, label_map).to(device)
        text_features, logits1, logits2, logits3, logits4 = model(
            visual_features, None, prompt_text, feat_lengths,
            DNP_use=False,
            snippet_text_features=snippet_text_batch
        )

        loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
        loss2 = CLASM(logits2, text_labels, feat_lengths, device)
        loss3 = text_contrast_loss(text_features)
        loss4 = CLASM_EVENT(logits3, text_labels, feat_lengths, device)
        loss5 = CLASM_BKG(logits4, text_labels, feat_lengths, device)
        loss = loss1 + loss2 + loss3 + loss4 + loss5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total1 += loss1.item()
        loss_total2 += loss2.item()

        step = i * normal_loader.batch_size * 2
        if step % 1280 == 0 and step != 0:
            print(f'teacher step={step} loss1={loss_total1/(i+1):.4f} loss2={loss_total2/(i+1):.4f}', flush=True)


def category_supervision_loss(category_logits, text_labels):
    targets = torch.argmax(text_labels.to(category_logits.device), dim=1)
    return F.cross_entropy(category_logits, targets)


def _forward_kwargs(args, snippet_text_batch=None):
    return {
        'snippet_text_features': snippet_text_batch,
        'use_motion_refine': args.use_motion_refine,
        'use_category_refine': args.use_category_refine,
        'prototype_temp': args.prototype_temp,
    }


def train_student(student, teacher, normal_loader, anomaly_loader, testloader, args, label_map, device):
    student.to(device)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    refiner_params, main_params = [], []
    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        if 'video_anomaly_refiner' in name:
            refiner_params.append(param)
        else:
            main_params.append(param)

    optimizer_main = torch.optim.AdamW(main_params, lr=args.lr)
    optimizer_refiner = StableAdamW(
        [{'params': refiner_params}], lr=args.lr,
        betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10
    )
    scheduler_main = MultiStepLR(optimizer_main, args.scheduler_milestones, args.scheduler_rate)
    total_iters_refiner = args.max_epoch * (len(normal_loader) + len(anomaly_loader))
    scheduler_refiner = WarmCosineScheduler(
        optimizer_refiner, base_value=args.lr, final_value=args.lr * 0.1,
        total_iters=total_iters_refiner, warmup_iters=100
    )

    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    auc1_best = -1.0
    map_best = -1.0
    auc1_ckpt_path = args.checkpoint_path.replace('.pth', '_best_auc1.pth') if args.checkpoint_path.endswith('.pth') else f'{args.checkpoint_path}_best_auc1'
    map_ckpt_path = args.checkpoint_path.replace('.pth', '_best_map.pth') if args.checkpoint_path.endswith('.pth') else f'{args.checkpoint_path}_best_map'
    DNP_use = getattr(args, 'DNP_use', True)

    for e in range(args.max_epoch):
        student.train()
        loss_total1 = loss_total2 = loss_total4 = loss_total5 = loss_kd = 0.0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)

        for i in range(min(len(normal_loader), len(anomaly_loader))):
            normal_features, normal_label, normal_lengths, normal_text_features, normal_snippet_text, normal_meta = _unpack_batch(
                next(normal_iter), args.use_snippet_text_gating, args.use_train_visual_retrieval
            )
            anomaly_features, anomaly_label, anomaly_lengths, anomaly_text_features, anomaly_snippet_text, anomaly_meta = _unpack_batch(
                next(anomaly_iter), args.use_snippet_text_gating, args.use_train_visual_retrieval
            )

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            raw_labels = list(normal_label) + list(anomaly_label)
            snippet_text_batch = None
            if args.use_snippet_text_gating:
                snippet_text_batch = torch.cat([normal_snippet_text, anomaly_snippet_text], dim=0).to(device)
            text_labels = get_batch_label(raw_labels, prompt_text, label_map).to(device)

            with torch.no_grad():
                t_result = teacher(
                    visual_features, None, prompt_text, feat_lengths,
                    DNP_use=False, return_visual_features=True, return_aux=True,
                    **_forward_kwargs(args, snippet_text_batch)
                )
                # (text, logits1, logits2, logits3, logits4, visual, aux)
                _, t_logits1, t_logits2, t_logits3, t_logits4, t_visual, t_aux = t_result

            s_result = student(
                visual_features, None, prompt_text, feat_lengths,
                return_visual_features=True, return_aux=True,
                **_forward_kwargs(args, snippet_text_batch)
            )
            if DNP_use:
                # (text, logits1, logits2, logits3, logits4, visual, aux, DNP)
                text_features, logits1, logits2, logits3, logits4, s_visual, s_aux, DNP = s_result
            else:
                # (text, logits1, logits2, logits3, logits4, visual, aux)
                text_features, logits1, logits2, logits3, logits4, s_visual, s_aux = s_result

            loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss3 = text_contrast_loss(text_features)
            loss4 = CLASM_EVENT(logits3, text_labels, feat_lengths, device)
            loss5 = CLASM_BKG(logits4, text_labels, feat_lengths, device)
            temporal_loss = temporal_consistency_loss(logits1, feat_lengths)
            event_shape_loss, event_trend_loss = event_completeness_distillation(logits1, t_logits1, feat_lengths)

            kd_bin = F.mse_loss(torch.sigmoid(logits1), torch.sigmoid(t_logits1))
            kd_multi = F.kl_div(
                F.log_softmax(logits2 / 2.0, dim=-1),
                F.softmax(t_logits2 / 2.0, dim=-1),
                reduction='batchmean'
            ) * 4.0
            kd_feat = F.mse_loss(F.normalize(s_visual, dim=-1), F.normalize(t_visual, dim=-1))
            kd_loss = 0.5 * kd_bin + 0.3 * kd_multi + 0.2 * kd_feat

            category_loss = logits1.new_tensor(0.0)
            if args.use_category_refine and 'category_logits' in s_aux:
                category_loss = category_supervision_loss(s_aux['category_logits'], text_labels)

            hard_neg_loss = logits1.new_tensor(0.0)
            if args.hard_neg_weight > 0:
                hard_neg_loss = hard_negative_margin_loss(
                    logits1, feat_lengths, text_labels,
                    topk=args.hard_neg_topk, margin=args.hard_neg_margin,
                )

            loss = (
                args.loss1_weight * loss1
                + args.loss2_weight * loss2
                + loss3
                + loss4
                + loss5
                + args.distill_weight * kd_loss
                + args.temporal_consistency_weight * temporal_loss
                + args.event_kd_weight * event_shape_loss
                + args.event_trend_weight * event_trend_loss
                + args.category_loss_weight * category_loss
                + args.hard_neg_weight * hard_neg_loss
            )

            if DNP_use:
                consist_loss = consistency_loss_fn(
                    logits1=logits1,
                    original_features=DNP['original_features'],
                    reconstructed_features=DNP['reconstructed_features'],
                    lengths=feat_lengths,
                )
                loss = loss + consist_loss + DNP['g_loss']

            optimizer_main.zero_grad()
            optimizer_refiner.zero_grad()
            loss.backward()
            optimizer_main.step()
            optimizer_refiner.step()
            scheduler_refiner.step()

            loss_total1 += loss1.item()
            loss_total2 += loss2.item()
            loss_total4 += loss4.item()
            loss_total5 += loss5.item()
            loss_kd += kd_loss.item()

            step = i * normal_loader.batch_size * 2
            if step % 1280 == 0 and step != 0:
                print(
                    f'epoch={e+1} step={step} loss1={loss_total1/(i+1):.4f} '
                    f'loss2={loss_total2/(i+1):.4f} loss4={loss_total4/(i+1):.4f} '
                    f'loss5={loss_total5/(i+1):.4f} kd={loss_kd/(i+1):.4f}',
                    flush=True
                )
                AUC, AP, avg_map = test(
                    student, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device,
                    use_snippet_text_gating=args.use_snippet_text_gating,
                    use_motion_refine=args.use_motion_refine,
                    use_category_refine=args.use_category_refine,
                    prototype_temp=args.prototype_temp,
                    gaussian_sigma=args.gaussian_sigma,
                    return_map=True,
                    DNP_use=DNP_use,
                    temp=getattr(args, 'temp', 5.0),
                )
                student.train()
                if AP > ap_best:
                    ap_best = AP
                    torch.save({'epoch': e, 'model_state_dict': student.state_dict(), 'ap': ap_best}, args.checkpoint_path)
                if AUC > auc1_best:
                    auc1_best = float(AUC)
                    torch.save({'epoch': e, 'model_state_dict': student.state_dict(),
                                'auc1': auc1_best, 'ap': float(AP), 'map': float(avg_map)}, auc1_ckpt_path)
                if avg_map > map_best:
                    map_best = float(avg_map)
                    torch.save({'epoch': e, 'model_state_dict': student.state_dict(),
                                'auc1': float(AUC), 'ap': float(AP), 'map': map_best}, map_ckpt_path)

        scheduler_main.step()
        torch.save(student.state_dict(), './ucf_cur_student.pth')
        print(f'epoch={e+1} final evaluation', flush=True)
        AUC, AP, avg_map = test(
            student, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device,
            use_snippet_text_gating=args.use_snippet_text_gating,
            use_motion_refine=args.use_motion_refine,
            use_category_refine=args.use_category_refine,
            prototype_temp=args.prototype_temp,
            gaussian_sigma=args.gaussian_sigma,
            return_map=True,
            DNP_use=DNP_use,
            temp=getattr(args, 'temp', 5.0),
        )
        student.train()
        if AP > ap_best:
            ap_best = AP
            torch.save({'epoch': e, 'model_state_dict': student.state_dict(), 'ap': ap_best}, args.checkpoint_path)
        if AUC > auc1_best:
            auc1_best = float(AUC)
            torch.save({'epoch': e, 'model_state_dict': student.state_dict(),
                        'auc1': auc1_best, 'ap': float(AP), 'map': float(avg_map)}, auc1_ckpt_path)
        if avg_map > map_best:
            map_best = float(avg_map)
            torch.save({'epoch': e, 'model_state_dict': student.state_dict(),
                        'auc1': float(AUC), 'ap': float(AP), 'map': map_best}, map_ckpt_path)

        if os.path.isfile(args.checkpoint_path):
            _try_restore_best(student, args.checkpoint_path)

    if not _try_restore_best(student, args.checkpoint_path):
        return
    torch.save(student.state_dict(), args.model_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(args=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args is None:
        args = ucf_option.parser.parse_args()
    args = ucf_option.resolve_text_feature_dirs(args)
    setup_seed(args.seed)
    log_file = _setup_logging(args.log_path)
    print(f'Text feature version: {args.text_feature_version}', flush=True)
    print(f'Text feature dir: {args.text_feature_dir}', flush=True)
    print(f'Snippet text feature dir: {args.snippet_text_feature_dir}', flush=True)

    label_map = {
        'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson',
        'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion',
        'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery',
        'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing',
        'Vandalism': 'vandalism'
    }

    normal_dataset = UCFDataset(
        args.visual_length, args.train_list, False, label_map, True,
        text_feature_dir=args.text_feature_dir,
        return_text_feature=True,
        snippet_text_feature_dir=args.snippet_text_feature_dir,
        return_snippet_text_feature=args.use_snippet_text_gating,
        return_metadata=args.use_train_visual_retrieval,
    )
    anomaly_dataset = UCFDataset(
        args.visual_length, args.train_list, False, label_map, False,
        text_feature_dir=args.text_feature_dir,
        return_text_feature=True,
        snippet_text_feature_dir=args.snippet_text_feature_dir,
        return_snippet_text_feature=args.use_snippet_text_gating,
        return_metadata=args.use_train_visual_retrieval,
    )
    test_dataset = UCFDataset(
        args.visual_length, args.test_list, True, label_map,
        snippet_text_feature_dir=args.snippet_text_feature_dir,
        return_snippet_text_feature=args.use_snippet_text_gating,
    )

    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    os.makedirs(os.path.dirname(args.model_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.teacher_model_path) or '.', exist_ok=True)

    teacher_path = args.teacher_model_path
    teacher_best_path = _teacher_best_path(teacher_path)
    args.retrieval_bank = build_train_retrieval_bank(normal_dataset, anomaly_dataset, args)
    teacher = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
                      args.visual_head, args.visual_layers, args.attn_window,
                      args.prompt_prefix, args.prompt_postfix, device,
                      dataset='ucf',
                      use_debiased_causal_graph=args.use_debiased_causal_graph,
                      debiased_graph_threshold=args.debiased_graph_threshold,
                      text_adapt_until=getattr(args, 'text_adapt_until', 3),
                      t_w=getattr(args, 't_w', 0.1),
                      num_prototypes=getattr(args, 'num_prototypes', 16),
                      decoder_depth=getattr(args, 'decoder_depth', 8),
                      normal_selection_ratio=getattr(args, 'normal_selection_ratio', 0.8))
    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    if args.train_teacher:
        _load_pretrained_if_available(teacher, args.pretrained_path)
        teacher_optimizer = torch.optim.AdamW(teacher.parameters(), lr=args.teacher_lr)
        ap_best = -1.0

        for te in range(args.teacher_epochs):
            run_teacher_epoch(teacher, normal_loader, anomaly_loader, teacher_optimizer,
                              prompt_text, label_map, device,
                              use_snippet_text_gating=args.use_snippet_text_gating,
                              retrieval_bank=args.retrieval_bank,
                              args=args)
            print(f'teacher epoch {te+1}/{args.teacher_epochs} done', flush=True)
            teacher_auc, teacher_ap, teacher_map = test(
                teacher, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device,
                use_snippet_text_gating=args.use_snippet_text_gating,
                use_motion_refine=args.use_motion_refine,
                use_category_refine=args.use_category_refine,
                prototype_temp=args.prototype_temp,
                gaussian_sigma=args.gaussian_sigma,
                return_map=True,
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

        if os.path.isfile(teacher_best_path) and _try_restore_best(teacher, teacher_best_path):
            print(f'Restored best teacher from {teacher_best_path} for distillation.', flush=True)
        torch.save(teacher.state_dict(), teacher_path)
    else:
        load_teacher_path = teacher_best_path if os.path.isfile(teacher_best_path) else teacher_path
        if not os.path.isfile(load_teacher_path):
            raise FileNotFoundError(f'Teacher checkpoint not found at {load_teacher_path}. Re-run with --train-teacher to create it.')
        print(f'Loading existing teacher from {load_teacher_path}...', flush=True)
        teacher_payload = _safe_torch_load(load_teacher_path, map_location='cpu')
        teacher_state = teacher_payload.get('model_state_dict') if isinstance(teacher_payload, dict) and 'model_state_dict' in teacher_payload else teacher_payload
        missing, unexpected = teacher.load_state_dict(teacher_state, strict=False)
        print(f'Loaded teacher with strict=False. Missing: {len(missing)}, Unexpected: {len(unexpected)}', flush=True)

    student = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
                      args.visual_head, args.visual_layers, args.attn_window,
                      args.prompt_prefix, args.prompt_postfix, device,
                      dataset='ucf',
                      use_debiased_causal_graph=args.use_debiased_causal_graph,
                      debiased_graph_threshold=args.debiased_graph_threshold,
                      text_adapt_until=getattr(args, 'text_adapt_until', 3),
                      t_w=getattr(args, 't_w', 0.1),
                      num_prototypes=getattr(args, 'num_prototypes', 16),
                      decoder_depth=getattr(args, 'decoder_depth', 8),
                      normal_selection_ratio=getattr(args, 'normal_selection_ratio', 0.8))
    _load_pretrained_if_available(student, args.pretrained_path)

    train_student(student, teacher, normal_loader, anomaly_loader, test_loader, args, label_map, device)
    log_file.close()


if __name__ == '__main__':
    main()

