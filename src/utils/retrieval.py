from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def pooled_visual_feature(path_value: str) -> np.ndarray:
    feature = np.load(path_value).astype(np.float32)
    pooled = feature if feature.ndim == 1 else feature.mean(axis=0)
    norm = np.linalg.norm(pooled) + 1e-6
    return pooled / norm


def label_to_class_index(label, label_map: dict) -> int:
    keys = list(label_map.keys())
    if label in keys:
        return keys.index(label)
    if isinstance(label, str) and '-' in label:
        first = label.split('-', 1)[0]
        if first in keys:
            return keys.index(first)
    return 0


@lru_cache(maxsize=8)
def build_visual_bank(csv_path: str, label_items: tuple[tuple[str, str], ...], max_items: int = 6000):
    path = Path(csv_path)
    if not csv_path or not path.exists():
        return None
    label_map = dict(label_items)
    df = pd.read_csv(path)
    if len(df) > max_items:
        df = df.iloc[np.linspace(0, len(df) - 1, max_items, dtype=np.int64)]
    visual_vectors = []
    class_indices = []
    anomaly_flags = []
    for _, row in df.iterrows():
        path_value = row['path']
        if not Path(path_value).exists():
            continue
        try:
            visual = pooled_visual_feature(path_value)
        except Exception:
            continue
        cls_idx = label_to_class_index(str(row['label']), label_map)
        visual_vectors.append(visual)
        class_indices.append(cls_idx)
        anomaly_flags.append(float(cls_idx != 0))
    if not visual_vectors:
        return None
    return {
        'visual': torch.from_numpy(np.stack(visual_vectors).astype(np.float32)),
        'class_indices': torch.tensor(class_indices, dtype=torch.long),
        'anomaly_flags': torch.tensor(anomaly_flags, dtype=torch.float32),
        'num_classes': len(label_map),
    }


class RetrievalBank:
    def __init__(self, train_list: str | None, label_map: dict, topk: int = 5, max_visual_bank: int = 6000):
        self.label_map = label_map
        self.topk = topk
        self.bank = build_visual_bank(train_list or '', tuple(label_map.items()), max_visual_bank)

    def query(self, visual_feature):
        num_classes = len(self.label_map)
        class_prior = torch.zeros(num_classes, dtype=torch.float32)
        anomaly_score = torch.zeros(1, dtype=torch.float32)
        confidence = torch.zeros(1, dtype=torch.float32)

        if self.bank is None:
            return class_prior, anomaly_score, confidence

        if isinstance(visual_feature, np.ndarray):
            query = torch.from_numpy(visual_feature.astype(np.float32))
        elif torch.is_tensor(visual_feature):
            query = visual_feature.detach().float().cpu()
        else:
            return class_prior, anomaly_score, confidence

        if query.dim() > 1:
            query = query.mean(dim=0)
        query = F.normalize(query, dim=-1)

        visual_bank = F.normalize(self.bank['visual'].float(), dim=-1)
        sim = visual_bank @ query
        k = min(max(int(self.topk), 1), sim.numel())
        values, indices = torch.topk(sim, k=k, largest=True)
        weights = torch.softmax(values, dim=0)
        selected_classes = self.bank['class_indices'][indices]
        selected_anomaly = self.bank['anomaly_flags'][indices]

        class_prior.scatter_add_(0, selected_classes, weights)
        if class_prior.sum() > 0:
            class_prior = class_prior / class_prior.sum().clamp_min(1e-6)
        anomaly_score[0] = torch.sum(weights * selected_anomaly)
        confidence[0] = values.mean().clamp(0.0, 1.0)
        return class_prior, anomaly_score, confidence
