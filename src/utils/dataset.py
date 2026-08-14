import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
from pathlib import Path
from functools import lru_cache
import utils.tools as tools
from utils.retrieval import RetrievalBank, pooled_visual_feature


@lru_cache(maxsize=8)
def _load_feature_lookup(feature_dir: str):
    lookup = {}
    path = Path(feature_dir)
    if not path.exists():
        return lookup
    for file in path.glob('*.npy'):
        lookup[file.stem] = str(file)
    return lookup


@lru_cache(maxsize=8)
def _load_snippet_feature_lookup(feature_dir: str):
    lookup = {}
    path = Path(feature_dir)
    if not path.exists():
        return lookup
    for file in path.glob('*.npy'):
        lookup[file.stem] = str(file)
    return lookup


def _canonical_video_id(raw: str) -> str:
    value = Path(str(raw)).stem
    value = value.replace('___', '__#')
    if '__' in value:
        prefix, suffix = value.split('__', 1)
        if prefix and suffix:
            value = suffix
    return value


def _base_video_id(raw: str) -> str:
    value = Path(str(raw)).stem
    if '__' in value:
        return value.split('__', 1)[0]
    return value


def _visual_key_from_path(visual_path: str):

    visual_name = Path(visual_path).stem
    base_name = visual_name.rsplit('__', 1)[0]
    if '__#' in base_name:
        video_name, clip_name = base_name.split('__#', 1)
        return f"{video_name}___{clip_name}"
    return base_name


def _audio_key_from_path(audio_path: Path):
    audio_name = audio_path.stem
    if '_' not in audio_name:
        return None
    return audio_name.split('_', 1)[1]


def _build_audio_index(audio_root: str):
    audio_index = {}
    for audio_path in Path(audio_root).glob('*.npy'):
        audio_key = _audio_key_from_path(audio_path)
        if audio_key is None:
            continue
        audio_index.setdefault(audio_key, []).append(str(audio_path))
    return {key: paths[0] for key, paths in audio_index.items() if len(paths) == 1}


class UCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict, normal: bool = False,
                 text_feature_dir: str | None = None, return_text_feature: bool = False,
                 snippet_text_feature_dir: str | None = None, return_snippet_text_feature: bool = False,
                 return_metadata: bool = False):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        self.return_text_feature = return_text_feature
        self.return_snippet_text_feature = return_snippet_text_feature
        self.return_metadata = return_metadata
        self.text_feature_lookup = _load_feature_lookup(text_feature_dir) if (return_text_feature and text_feature_dir) else {}
        self.snippet_text_feature_lookup = _load_snippet_feature_lookup(snippet_text_feature_dir) if (return_snippet_text_feature and snippet_text_feature_dir) else {}
        if normal == True and test_mode == False:
            self.df = self.df.loc[self.df['label'] == 'Normal']
            self.df = self.df.reset_index()
        elif test_mode == False:
            self.df = self.df.loc[self.df['label'] != 'Normal']
            self.df = self.df.reset_index()

    def __len__(self):
        return self.df.shape[0]

    def _get_text_feature(self, path_value):
        if not self.return_text_feature:
            return None
        key = _canonical_video_id(path_value)
        feature_path = self.text_feature_lookup.get(key)
        if feature_path is None:
            return torch.zeros(512, dtype=torch.float32)
        feature = np.load(feature_path).astype(np.float32)
        return torch.from_numpy(feature)

    def _get_snippet_text_feature(self, path_value, target_length: int):
        if not self.return_snippet_text_feature:
            return None
        key = _base_video_id(path_value)
        feature_path = self.snippet_text_feature_lookup.get(key)
        if feature_path is None:
            return torch.zeros(target_length, 512, dtype=torch.float32)
        feature = np.load(feature_path).astype(np.float32)
        feature, _ = tools.process_feat(feature, target_length)
        return torch.from_numpy(feature)

    def __getitem__(self, index):
        clip_path = self.df.loc[index]['path']
        clip_feature = np.load(clip_path)
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        outputs = [clip_feature, clip_label, clip_length]
        if self.return_text_feature:
            outputs.append(self._get_text_feature(clip_path))
        if self.return_snippet_text_feature:
            outputs.append(self._get_snippet_text_feature(clip_path, self.clip_dim))
        if self.return_metadata:
            outputs.append(_canonical_video_id(clip_path))
        return tuple(outputs)


class XDDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict, use_audio: bool = False,
                 audio_root: str = '', text_feature_dir: str | None = None, return_text_feature: bool = False,
                 snippet_text_feature_dir: str | None = None, return_snippet_text_feature: bool = False,
                 return_metadata: bool = False, use_rag: bool = False, rag_topk: int = 5,
                 rag_max_bank_size: int = 6000, rag_train_list: str | None = None):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.use_audio = use_audio
        self.audio_root = audio_root
        self.return_text_feature = return_text_feature
        self.return_snippet_text_feature = return_snippet_text_feature
        self.return_metadata = return_metadata
        self.use_rag = use_rag
        self.text_feature_lookup = _load_feature_lookup(text_feature_dir) if (return_text_feature and text_feature_dir) else {}
        self.snippet_text_feature_lookup = _load_snippet_feature_lookup(snippet_text_feature_dir) if (return_snippet_text_feature and snippet_text_feature_dir) else {}
        self.retrieval_bank = RetrievalBank(
            rag_train_list or file_path,
            label_map,
            topk=rag_topk,
            max_visual_bank=rag_max_bank_size,
        ) if use_rag else None

        if self.use_audio:
            audio_index = _build_audio_index(self.audio_root)
            self.df['original_index'] = self.df.index
            self.df['video_key'] = self.df['path'].map(_visual_key_from_path)
            self.df['audio_path'] = self.df['video_key'].map(audio_index)
            self.df = self.df[self.df['path'].map(lambda p: Path(p).exists())]
            self.df = self.df[self.df['audio_path'].notna()]
            self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return self.df.shape[0]

    def _get_text_feature(self, path_value):
        if not self.return_text_feature:
            return None
        key = _canonical_video_id(path_value)
        feature_path = self.text_feature_lookup.get(key)
        if feature_path is None:
            return torch.zeros(512, dtype=torch.float32)
        feature = np.load(feature_path).astype(np.float32)
        return torch.from_numpy(feature)

    def _get_snippet_text_feature(self, path_value, target_length: int):
        if not self.return_snippet_text_feature:
            return None
        key = _base_video_id(path_value)
        feature_path = self.snippet_text_feature_lookup.get(key)
        if feature_path is None:
            return torch.zeros(target_length, 512, dtype=torch.float32)
        feature = np.load(feature_path).astype(np.float32)
        feature, _ = tools.process_feat(feature, target_length)
        return torch.from_numpy(feature)

    def __getitem__(self, index):
        row = self.df.loc[index]
        clip_path = row['path']
        clip_feature = np.load(clip_path)
        clip_feature_length = clip_feature.shape[0]
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = row['label']

        outputs = [clip_feature]
        if self.use_audio:
            audio_feature = np.load(row['audio_path']).astype(np.float32)
            if self.test_mode == False:
                target_len = clip_feature.shape[0]
                audio_feature = tools.resample_feat(audio_feature, target_len)
            else:
                audio_feature = tools.resample_feat(audio_feature, clip_feature_length)
                audio_feature, _ = tools.split_by_length(audio_feature, self.clip_dim)
            audio_feature = torch.tensor(audio_feature)
            outputs.append(audio_feature)

        outputs.extend([clip_label, clip_length])
        if self.return_text_feature:
            outputs.append(self._get_text_feature(clip_path))
        if self.return_snippet_text_feature:
            outputs.append(self._get_snippet_text_feature(clip_path, self.clip_dim))
        if self.use_rag:
            rag_query = pooled_visual_feature(clip_path)
            rag_class_prior, rag_anomaly_score, rag_confidence = self.retrieval_bank.query(rag_query)
            outputs.extend([rag_class_prior, rag_anomaly_score, rag_confidence])
        if self.return_metadata:
            outputs.append(_canonical_video_id(clip_path))
        if self.use_audio:
            outputs.append(int(row['original_index']))
        return tuple(outputs)
