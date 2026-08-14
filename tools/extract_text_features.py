#!/usr/bin/env python
"""从原始文本 JSONL 提取 CLIP ViT-B/16 文本特征。

原始文本数据（JSONL）里每个视频有 global_context + segments（LLM 生成的客观动作描述）。
本脚本用 CLIP 文本编码器把它们编码成 512 维特征，文件命名与 data/ 里现有特征一致
（video_id 里的 '___' 换成 '__#'）。

用法:
  # 视频级（每个视频一个 512 维，mean-pool global_context + 所有 segment）
  python tools/extract_text_features.py --input text_data/data_ucf1.jsonl \
      --output data/ucf_text --mode video

  # snippet 级（每个 segment 一个 512 维 → N×512）
  python tools/extract_text_features.py \
      --input text_data/data_ucf.timestamped.llm.jsonl \
      --output data/ucf_text_llm_snippets --mode snippet
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))
from clip import clip  # noqa: E402


def _feature_name(video_id: str) -> str:
    return str(video_id).replace('___', '__#')


def _texts_from_record(obj: dict, mode: str) -> list:
    texts = []
    if mode == 'video':
        gc = (obj.get('global_context') or '').strip()
        if gc:
            texts.append(gc)
    for seg in obj.get('segments') or []:
        t = (seg.get('text') or '').strip()
        if t:
            texts.append(t)
    return texts


@torch.no_grad()
def _encode(model, device, texts, mode):
    if not texts:
        return (np.zeros(512, dtype=np.float32) if mode == 'video'
                else np.zeros((0, 512), dtype=np.float32))
    tokens = clip.tokenize(texts, truncate=True).to(device)
    emb = model.encode_token(tokens)
    feats = model.encode_text(emb, tokens)
    feats = F.normalize(feats.float(), dim=-1)
    if mode == 'video':
        pooled = F.normalize(feats.mean(dim=0), dim=0)
        return pooled.cpu().numpy().astype(np.float32)
    return feats.cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--mode', choices=['video', 'snippet'], required=True)
    ap.add_argument('--model', default='ViT-B/16')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = clip.load(args.model, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = written = empty = 0
    with open(args.input, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            total += 1
            texts = _texts_from_record(obj, args.mode)
            if not texts:
                empty += 1
            feat = _encode(model, device, texts, args.mode)
            name = _feature_name(obj['video_id'])
            np.save(out_dir / f'{name}.npy', feat)
            written += 1
            if written % 500 == 0:
                print(f'written={written} empty={empty}', flush=True)

    print(f'done total={total} written={written} empty={empty} -> {out_dir}', flush=True)


if __name__ == '__main__':
    main()
