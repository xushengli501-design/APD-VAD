from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj
from utils.adapter_modules import SimpleAdapter, SimpleProj
from utils.dnp_vision_transformer import Aggregation_Block, Prototype_Block

class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class AudioCrossAttention(nn.Module):
    def __init__(self, visual_dim: int, audio_dim: int, num_heads: int):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, visual_dim)
        self.visual_norm = LayerNorm(visual_dim)
        self.audio_norm = LayerNorm(visual_dim)
        self.cross_attn = nn.MultiheadAttention(visual_dim, num_heads, batch_first=True)
        self.gate = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_dim * 2, visual_dim)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_dim, visual_dim))
        ]))

    def forward(self, visual_features: torch.Tensor, audio_features: torch.Tensor):
        audio_proj = self.audio_proj(audio_features)
        visual_norm = self.visual_norm(visual_features)
        audio_norm = self.audio_norm(audio_proj)
        attn_output, _ = self.cross_attn(visual_norm, audio_norm, audio_norm, need_weights=False)
        gate = torch.sigmoid(self.gate(torch.cat([visual_features, attn_output], dim=-1)))
        return visual_features + gate * attn_output


class CLIP_Adapter(nn.Module):
    def __init__(self, clipmodel, device, text_adapt_until=3, t_w=0.1):
        super().__init__()
        self.clipmodel = clipmodel
        self.text_adapt_until = text_adapt_until
        self.t_w = t_w
        self.device = device
        self.text_adapter = nn.ModuleList(
            [SimpleAdapter(512, 512) for _ in range(text_adapt_until)] +
            [SimpleProj(512, 512, relu=True)]
        )
        for p in self.text_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_text(self, text):
        cast_dtype = self.clipmodel.token_embedding.weight.dtype
        x = self.clipmodel.token_embedding(text).to(cast_dtype)
        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)
        for i in range(len(self.clipmodel.transformer.resblocks)):
            x = self.clipmodel.transformer.resblocks[i](x)
            if i < self.text_adapt_until:
                adapt_out = self.text_adapter[i](x)
                adapt_out = adapt_out * x.norm(dim=-1, keepdim=True) / (adapt_out.norm(dim=-1, keepdim=True) + 1e-6)
                x = self.t_w * adapt_out + (1 - self.t_w) * x
        x = x.permute(1, 0, 2)
        x = self.clipmodel.ln_final(x)
        eot_indices = text.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot_indices]
        x = self.text_adapter[-1](x)
        return x


class SGNM(nn.Module):
    def __init__(self, feature_dim=512, num_prototypes=16, num_heads=8,
                 extractor_depth=1, decoder_depth=8, normal_selection_ratio=0.8):
        super().__init__()
        self.normal_selection_ratio = normal_selection_ratio
        self.video_prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        self.dnp_extractor = nn.ModuleList([
            Aggregation_Block(dim=feature_dim, num_heads=num_heads, mlp_ratio=4.,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
            for _ in range(extractor_depth)
        ])
        self.bottleneck = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(feature_dim, feature_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(feature_dim * 4, feature_dim))
        ]))
        self.decoder = nn.ModuleList(
            [Prototype_Block(dim=feature_dim, num_heads=num_heads, mlp_ratio=4.,
                             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                             no_residual=True)] +
            [Prototype_Block(dim=feature_dim, num_heads=num_heads, mlp_ratio=4.,
                             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                             no_residual=False)
             for _ in range(decoder_depth - 1)]
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def gather_loss(self, query, keys):
        dist = 1.0 - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        return dist.min(dim=2).values.mean()

    def forward(self, visual_features, logits1):
        B, N, D = visual_features.shape
        with torch.no_grad():
            scores = torch.sigmoid(logits1.squeeze(-1))
            num_normal = max(1, int(N * self.normal_selection_ratio))
            _, idx = torch.topk(scores, k=N, largest=False, dim=1)
            normal_idx = idx[:, :num_normal]
            normal_feats = torch.gather(visual_features, 1,
                                        normal_idx.unsqueeze(-1).expand(-1, -1, D))
        proto = self.video_prototypes.unsqueeze(0).expand(B, -1, -1)
        for blk in self.dnp_extractor:
            proto = blk(proto, normal_feats)
        g_loss = self.gather_loss(normal_feats, proto)
        query = self.bottleneck(visual_features)
        recon = query
        for blk in self.decoder:
            recon = blk(recon, proto)
        return recon, g_loss


class CLIPVAD(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 device,
                 audio_dim: int = 128,
                 audio_fusion_mode: str = 'normal',
                 audio_cross_attn_heads: int = 4,
                 use_debiased_causal_graph: bool = False,
                 debiased_graph_threshold: float = 0.7,
                 causal_repr_alpha: float = 0.2,
                 causal_repr_detach: bool = False,
                 snippet_gate_temperature: float = 0.1,
                 snippet_gate_residual: float = 1.0,
                 use_rag: bool = False,
                 rag_weight: float = 0.05,
                 rag_conf_gate: bool = False,
                 use_dnp: bool = False,
                 dnp_num_prototypes: int = 16,
                 dnp_decoder_depth: int = 8,
                 dnp_normal_selection_ratio: float = 0.8,
                 use_clip_adapter: bool = False,
                 clip_adapter_layers: int = 3,
                 clip_adapter_weight: float = 0.1):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device
        self.use_debiased_causal_graph = use_debiased_causal_graph
        self.debiased_graph_threshold = debiased_graph_threshold
        self.causal_repr_alpha = causal_repr_alpha
        self.causal_repr_detach = causal_repr_detach
        self.snippet_gate_temperature = snippet_gate_temperature
        self.snippet_gate_residual = snippet_gate_residual
        self.use_rag = use_rag
        self.rag_weight = rag_weight
        self.rag_conf_gate = rag_conf_gate
        self.use_dnp = use_dnp
        self.use_clip_adapter = use_clip_adapter

        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.causal_repr_proj = nn.Linear(width, visual_width)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)
        self.rag_proj = nn.Linear(num_class + 2, 1)
        self.text_gate_scale = nn.Parameter(torch.tensor(1.0))
        self.motion_refine = None
        self.audio_dim = audio_dim
        self.audio_fusion_mode = audio_fusion_mode
        self.audio_projection = nn.Linear(audio_dim, self.visual_width)
        self.audio_gate = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(self.visual_width * 2, self.visual_width)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(self.visual_width, self.visual_width))
        ]))
        self.audio_cross_attention = AudioCrossAttention(self.visual_width, audio_dim, audio_cross_attn_heads)

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.clip_adapter = CLIP_Adapter(self.clipmodel, device, clip_adapter_layers, clip_adapter_weight) if use_clip_adapter else None
        self.video_anomaly_refiner = SGNM(
            feature_dim=visual_width,
            num_prototypes=dnp_num_prototypes,
            num_heads=8,
            extractor_depth=1,
            decoder_depth=dnp_decoder_depth,
            normal_selection_ratio=dnp_normal_selection_ratio,
        ) if use_dnp else None
        self._text_features_cache = None

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0

        return mask

    def _masked_temporal_mean(self, x, seq_len):
        if seq_len is None:
            return x.mean(dim=1, keepdim=True)
        mean = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)
        for i, length in enumerate(seq_len):
            valid_length = int(length.item()) if torch.is_tensor(length) else int(length)
            if valid_length <= 0:
                continue
            mean[i] = x[i, :valid_length].mean(dim=0, keepdim=True)
        return mean

    def _build_causal_mask(self, length, device, dtype):
        return torch.tril(torch.ones(length, length, device=device, dtype=dtype))

    def _dfs_components(self, adj_bin):
        """adj_bin: T×T bool numpy array. Returns list of component index lists."""
        T = adj_bin.shape[0]
        visited = [False] * T
        components = []
        for start in range(T):
            if visited[start]:
                continue
            stack = [start]
            comp = []
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                comp.append(node)
                for nb in range(T):
                    if adj_bin[node, nb] and not visited[nb]:
                        stack.append(nb)
            components.append(comp)
        return components

    def build_acc_pseudo_labels(self, x, text_sim_scores, lengths, text_features_ori, eta=0.5, threshold=0.9):
        """
        Build ACC dense pseudo-labels for teacher distillation (LAS-VAD style).

        x: B×T×D visual features (post encode_video)
        text_sim_scores: B×T×C  (logits2.softmax, q^l from teacher)
        lengths: list/tensor of valid lengths per sample
        text_features_ori: unused, kept for API compatibility
        eta: text-correction strength
        threshold: binarization threshold (LAS-VAD uses 0.9)

        Returns g_teacher: B×T×C  — each frame gets the averaged q^l of its component (hard pseudo-label)
        """
        B, T, D = x.shape
        C = text_sim_scores.shape[-1]
        g = torch.zeros(B, T, C, device=x.device, dtype=x.dtype)

        x_norm = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        A_v = torch.bmm(x_norm, x_norm.permute(0, 2, 1))  # B×T×T

        for b in range(B):
            valid_len = int(lengths[b].item()) if torch.is_tensor(lengths[b]) else int(lengths[b])
            if valid_len <= 0:
                continue

            q = text_sim_scores[b, :valid_len]  # T'×C  (already softmax probabilities)

            # Text correction: max_c min(q[i,c], q[j,c])  — LAS-VAD eq.(13)
            q_i = q.unsqueeze(1).expand(-1, valid_len, -1)   # T'×T'×C
            q_j = q.unsqueeze(0).expand(valid_len, -1, -1)   # T'×T'×C
            text_corr = torch.min(q_i, q_j).max(dim=-1).values  # T'×T'

            A_hat = A_v[b, :valid_len, :valid_len] * (1.0 + eta * text_corr)

            # Causal mask (lower-triangular) for debiasing
            causal_mask = self._build_causal_mask(valid_len, x.device, x.dtype)
            A_hat = A_hat * causal_mask

            # Binarize and DFS
            adj_bin = (A_hat > threshold).cpu().numpy()
            components = self._dfs_components(adj_bin)

            # Each component's pseudo-label = average of member frames' q^l (LAS-VAD: averaged logits)
            for comp in components:
                proto_label = q[comp].mean(dim=0)  # C  — averaged softmax probs within component
                for t in comp:
                    g[b, t] = proto_label

        return g

    def adj4_debiased(self, x, seq_len):
        soft = nn.Softmax(1)
        centered = x - self._masked_temporal_mean(x, seq_len)
        x2 = centered.matmul(centered.permute(0, 2, 1))
        x_norm = torch.norm(centered, p=2, dim=2, keepdim=True)
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2 / (x_norm_x + 1e-20)
        output = torch.zeros_like(x2)
        threshold = self.debiased_graph_threshold
        if seq_len is None:
            for i in range(centered.shape[0]):
                length = centered.shape[1]
                causal_mask = self._build_causal_mask(length, centered.device, centered.dtype)
                adj2 = x2[i] * causal_mask
                adj2 = F.threshold(adj2, threshold, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                valid_length = int(seq_len[i].item()) if torch.is_tensor(seq_len[i]) else int(seq_len[i])
                if valid_length <= 0:
                    continue
                causal_mask = self._build_causal_mask(valid_length, centered.device, centered.dtype)
                adj2 = x2[i, :valid_length, :valid_length] * causal_mask
                adj2 = F.threshold(adj2, threshold, 0)
                adj2 = soft(adj2)
                output[i, :valid_length, :valid_length] = adj2
        return output

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1)) # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def encode_video(self, images, padding_mask, lengths, override_use_debiased_causal_graph=None):

        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        x, _ = self.temporal((images, None))
        x = x.permute(1, 0, 2)

        base_adj = self.adj4(x, lengths)
        causal_adj = self.adj4_debiased(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])

        use_causal_graph = self.use_debiased_causal_graph if override_use_debiased_causal_graph is None else override_use_debiased_causal_graph

        if use_causal_graph:
            causal_input = x.detach() if self.causal_repr_detach else x
            causal_h = self.gelu(self.gc1(causal_input, causal_adj))
            causal_repr = self.gelu(self.gc2(causal_h, causal_adj))
            causal_repr = self.causal_repr_proj(causal_repr)
            x_for_graph = x + self.causal_repr_alpha * causal_repr
        else:
            x_for_graph = x

        x1_h = self.gelu(self.gc1(x_for_graph, base_adj))
        x2_h = self.gelu(self.gc3(x_for_graph, disadj))

        x1 = self.gelu(self.gc2(x1_h, base_adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)
        x = self.linear(x)

        return x

    def encode_textprompt(self, text):
        if self.use_clip_adapter and self.clip_adapter is not None:
            tokens = clip.tokenize(text).to(self.device)
            return self.clip_adapter.encode_text(tokens)
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)

        return text_features

    def apply_motion_refinement(self, visual_features):
        if self.motion_refine is None:
            self.motion_refine = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(self.visual_width * 2, self.visual_width)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(self.visual_width, self.visual_width))
            ])).to(device=visual_features.device, dtype=visual_features.dtype)
        delta = visual_features[:, 1:] - visual_features[:, :-1]
        delta = torch.cat([torch.zeros_like(visual_features[:, :1]), delta], dim=1)
        motion_gate = torch.sigmoid(self.motion_refine(torch.cat([visual_features, delta], dim=-1)))
        return visual_features + motion_gate * delta

    def apply_audio_fusion(self, visual_features, audio_features):
        audio_features = audio_features.to(device=visual_features.device, dtype=visual_features.dtype)
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)
        if audio_features.dim() == 4:
            audio_features = audio_features.reshape(-1, audio_features.shape[-2], audio_features.shape[-1])
        assert audio_features.shape[-1] == self.audio_dim, (
            f"audio feature dim mismatch: audio={audio_features.shape[-1]}, expected={self.audio_dim}"
        )
        assert audio_features.shape[:2] == visual_features.shape[:2], (
            f"audio/visual shape mismatch: audio={audio_features.shape}, visual={visual_features.shape}"
        )
        if self.audio_fusion_mode == 'identity':
            return visual_features
        if self.audio_fusion_mode == 'cross_attn':
            return self.audio_cross_attention(visual_features, audio_features)
        audio_proj = self.audio_projection(audio_features)
        audio_gate = torch.sigmoid(self.audio_gate(torch.cat([visual_features, audio_proj], dim=-1)))
        if self.audio_fusion_mode == 'stats':
            visual_norm = visual_features.norm(dim=-1).mean().item()
            audio_norm = audio_proj.norm(dim=-1).mean().item()
            fused = visual_features + audio_gate * audio_proj
            fused_norm = fused.norm(dim=-1).mean().item()
            gate_mean = audio_gate.mean().item()
            gate_min = audio_gate.min().item()
            gate_max = audio_gate.max().item()
            print(
                f"audio_fusion_stats: visual_norm={visual_norm:.6f} audio_norm={audio_norm:.6f} fused_norm={fused_norm:.6f} gate_mean={gate_mean:.6f} gate_min={gate_min:.6f} gate_max={gate_max:.6f}"
            )
            return fused
        return visual_features + audio_gate * audio_proj

    def forward(self, visual, padding_mask, text, lengths, use_motion_refine=False, audio=None, use_audio_aux=False,
                text_features_override=None, return_visual_features=False, snippet_text_features=None, return_aux=False,
                rag_features=None, force_visual_only=False, classification_on_pure_visual=False,
                override_use_debiased_causal_graph=None, return_acc_pseudo_labels=False,
                acc_eta=0.5, acc_threshold=0.7, use_dnp=None, use_dcsa=False):
        visual_features_pure = self.encode_video(
            visual, padding_mask, lengths,
            override_use_debiased_causal_graph=override_use_debiased_causal_graph,
        )
        if use_motion_refine:
            visual_features_pure = self.apply_motion_refinement(visual_features_pure)
        visual_features_fused = visual_features_pure
        use_audio_path = use_audio_aux and audio is not None and not force_visual_only
        if use_audio_path:
            visual_features_fused = self.apply_audio_fusion(visual_features_pure, audio)
        classification_features = visual_features_pure if classification_on_pure_visual else visual_features_fused
        logits1 = self.classifier(classification_features + self.mlp2(classification_features))
        aux = {
            'used_audio_aux': use_audio_path,
            'used_snippet_gate': False,
            'used_rag': False,
            'used_causal_graph': self.use_debiased_causal_graph if override_use_debiased_causal_graph is None else override_use_debiased_causal_graph,
        }
        use_rag_path = self.use_rag and rag_features is not None and not force_visual_only
        if use_rag_path:
            rag_features = rag_features.to(device=classification_features.device, dtype=classification_features.dtype)
            rag_bias = torch.tanh(self.rag_proj(rag_features)).unsqueeze(1)
            if self.rag_conf_gate:
                rag_confidence = rag_features[:, -1:].unsqueeze(1)
                rag_bias = rag_bias * rag_confidence
            logits1 = logits1 + self.rag_weight * rag_bias
            aux['used_rag'] = True
        if snippet_text_features is not None and not force_visual_only:
            snippet_text_features = snippet_text_features.to(device=classification_features.device, dtype=classification_features.dtype)
            visual_norm = classification_features / classification_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            snippet_text_norm = snippet_text_features / snippet_text_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            text_prior = torch.sigmoid((visual_norm * snippet_text_norm).sum(dim=-1, keepdim=True) / self.snippet_gate_temperature)
            logits1 = logits1 + self.text_gate_scale.sigmoid() * self.snippet_gate_residual * text_prior
            aux['text_prior'] = text_prior
            aux['used_snippet_gate'] = True

        if text_features_override is None:
            text_features_ori = self.encode_textprompt(text)
        else:
            text_features_ori = text_features_override

        logits_attn = logits1.permute(0, 2, 1)
        visual_attn = logits_attn @ visual_features_pure
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        if text_features_ori.dim() == 2:
            visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])
            text_features = text_features_ori.unsqueeze(0)
            text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
        else:
            visual_attn = visual_attn.expand(-1, text_features_ori.shape[1], -1)
            text_features = text_features_ori
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)

        visual_features_norm = visual_features_pure / visual_features_pure.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        text_features_norm = text_features_norm.permute(0, 2, 1)
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07

        outputs = [text_features_ori, logits1, logits2]

        # DCSA: decouple event-centric and background-centric prototypes → logits3, logits4
        if use_dcsa:
            s_det = torch.sigmoid(logits1)
            w_event = torch.softmax(s_det.squeeze(-1), dim=1).unsqueeze(-1)
            w_bkg = (1.0 - w_event) / (1.0 - w_event).sum(dim=1, keepdim=True).clamp_min(1e-6)
            abn_feat = torch.matmul(w_event.permute(0, 2, 1), visual_features_pure)
            nor_feat = torch.matmul(w_bkg.permute(0, 2, 1), visual_features_pure)
            if text_features_ori.dim() == 2:
                nor_text = text_features_ori.unsqueeze(0).expand(abn_feat.shape[0], -1, -1)
            else:
                nor_text = text_features_ori
            nor_text_norm = F.normalize(nor_text, dim=-1).permute(0, 2, 1)
            logits3 = F.normalize(abn_feat, dim=-1) @ nor_text_norm.type(abn_feat.dtype) / 0.07
            logits4 = F.normalize(nor_feat, dim=-1) @ nor_text_norm.type(nor_feat.dtype) / 0.07
            outputs = [text_features_ori, logits1, logits2, logits3, logits4]

        # DNP: dynamic normal prototype reconstruction
        _use_dnp = self.use_dnp if use_dnp is None else use_dnp
        dnp_dict = None
        if _use_dnp and self.video_anomaly_refiner is not None:
            recon_features, g_loss = self.video_anomaly_refiner(visual_features_pure, logits1)
            dnp_dict = {
                'reconstructed_features': recon_features,
                'original_features': visual_features_pure,
                'g_loss': g_loss,
            }
            outputs = list(outputs) + [dnp_dict]
        if return_visual_features:
            outputs.append(visual_features_pure)
        if return_acc_pseudo_labels:
            with torch.no_grad():
                q_teacher = logits2.softmax(dim=-1).detach()
                # text_features_ori is C×D (2D) or B×C×D (3D); pass as-is
                g_teacher = self.build_acc_pseudo_labels(
                    visual_features_pure.detach(), q_teacher, lengths,
                    text_features_ori.detach(),
                    eta=acc_eta, threshold=acc_threshold,
                )
            outputs.append(g_teacher)
        if return_aux:
            outputs.append(aux)
        return tuple(outputs)
    