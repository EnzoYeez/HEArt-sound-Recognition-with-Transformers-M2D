import logging 
import torch
import numpy as np
from pathlib import Path
from functools import partial
from einops import rearrange
import nnAudio.features
import timm

# ============================
# Basic Configuration Class
# ============================
class M2DConfig:
    def __init__(self):
        self.weight_file = ''
        self.feature_d = 768 * 5
        self.norm_type = all
        self.pooling_type = 'mean'
        self.model = ''
        self.input_size = [80, 208]
        self.patch_size = [16, 16]
        self.sr = '16k'
        self.flat_features = False

def resolve_size(sz):
    return [sz, sz] if isinstance(sz, int) else sz

# ============================
# Patch Embedding Module
# ============================
class PatchEmbedding(torch.nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = resolve_size(img_size)
        patch_size = resolve_size(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_h, self.grid_w = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.grid_h * self.grid_w
        self.flatten = flatten
        self.proj = torch.nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else torch.nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        return self.norm(x.flatten(2).transpose(1, 2)) if self.flatten else self.norm(x)

# ============================
# Backbone Network
# ============================
class LocalViT(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patch_embed = PatchEmbedding(self.patch_embed.img_size, self.patch_embed.patch_size,
                                          self.patch_embed.proj.in_channels, self.patch_embed.proj.out_channels)
        self.norm_stats = torch.nn.Parameter(torch.tensor([-7.1, 4.2]), requires_grad=False)
        del self.head

    def patch_size(self):
        return np.array(self.patch_embed.patch_size)

    def grid_size(self):
        return np.array(self.patch_embed.img_size) // self.patch_size()

    def forward_encoder(self, x):
        x = self.patch_embed(x)
        pos_embed = self.pos_embed[:, 1:, :]
        if x.size(1) < pos_embed.size(1):
            d = pos_embed.size(-1)
            f = self.grid_size()[0]
            t = x.size(1) // f
            pos_embed = pos_embed.view(1, f, -1, d)[:, :, :t, :].reshape(1, f * t, d)
        x = x + pos_embed
        cls = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat((cls.expand(x.size(0), -1, -1), x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

# ============================
# Utility Functions
# ============================
def parse_model_info(name):
    base, *params = name.split('-')
    input_sz = list(map(int, params[0].split('x')))
    patch_sz = list(map(int, params[1].split('x')))
    sample_rate = params[2] if len(params) > 2 else '16k'
    return input_sz, patch_sz, sample_rate, base

def filter_valid_weights(model, ckpt, fname):
    model_keys = [n for n, _ in model.named_parameters()]
    kept, removed = {}, []
    for k in ckpt:
        if k in model_keys:
            kept[k] = ckpt[k]
        else:
            removed.append(k)
    print(f" using {len(kept)} weights, dropped {len(ckpt) - len(kept)} from {Path(fname)}")
    return kept

def load_head_weights(ckpt, norm_layer, linear_head):
    if 'module.head.norm.running_mean' in ckpt:
        norm_layer.load_state_dict({
            'running_mean': ckpt['module.head.norm.running_mean'],
            'running_var': ckpt['module.head.norm.running_var']
        })
        linear_head.load_state_dict({
            'weight': ckpt['module.head.mlp.mlp.0.weight'],
            'bias': ckpt['module.head.mlp.mlp.0.bias']
        })

def convert_ckpt_keys(ckpt):
    return {k.replace('module.ar.runtime.backbone.', ''): v for k, v in ckpt.items()}

def build_clap_modules(model, ckpt):
    if 'audio_proj.0.weight' in ckpt:
        dim = ckpt['audio_proj.0.weight'].shape[1]
        model.audio_proj = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, dim)
        )
        model.text_proj = torch.nn.Linear(ckpt['text_proj.weight'].shape[1], ckpt['text_proj.weight'].shape[0]) \
            if 'text_proj.weight' in ckpt else torch.nn.Identity()

def init_backbone(cfg, weight_path):
    cfg.input_size, cfg.patch_size, cfg.sr, cfg.model = parse_model_info(Path(weight_path).parent.name)
    raw_ckpt = torch.load(weight_path, map_location='cpu')
    ckpt = convert_ckpt_keys(raw_ckpt.get('model', raw_ckpt))
    if 'norm_stats' not in ckpt:
        ckpt['norm_stats'] = torch.tensor([-7.1, 4.2])
    model = LocalViT(
        in_chans=1, img_size=cfg.input_size, patch_size=cfg.patch_size, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)
    )
    build_clap_modules(model, ckpt)
    valid_ckpt = filter_valid_weights(model, ckpt, weight_path)
    msg = model.load_state_dict(valid_ckpt)
    print(msg); logging.info(msg)
    cfg.mean, cfg.std = model.state_dict()['norm_stats'].cpu().numpy()
    return model.eval(), ckpt

def create_mel_transform(cfg):
    if cfg.sr == '16k':
        cfg.sample_rate, cfg.n_fft, cfg.window_size, cfg.hop_size = 16000, 400, 400, 160
        cfg.n_mels, cfg.f_min, cfg.f_max = 80, 50, 8000
    elif cfg.sr == '32k':
        cfg.sample_rate, cfg.n_fft, cfg.window_size, cfg.hop_size = 32000, 800, 800, 320
        cfg.n_mels, cfg.f_min, cfg.f_max = 80, 50, 16000
    else:
        raise ValueError(f"Unsupported sampling rate: {cfg.sr}")
    return nnAudio.features.MelSpectrogram(
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        win_length=cfg.window_size,
        hop_length=cfg.hop_size,
        n_mels=cfg.n_mels,
        fmin=cfg.f_min,
        fmax=cfg.f_max,
        power=2,
        verbose=False,
        center=True,
    )

def compute_timestamps(cfg, batch_audio, frames):
    dur_sec = len(batch_audio[0]) / cfg.sample_rate
    steps = dur_sec / frames.size(1) * 1000
    base = torch.arange(frames.size(1)).float() * steps
    return base.unsqueeze(0).repeat(len(batch_audio), 1)

# ============================
# Main Model
# ============================
class PortableM2D(torch.nn.Module):
    def __init__(self, weight_file, num_classes=None, freeze_embed=False, flat_features=None):
        super().__init__()
        self.cfg = M2DConfig()
        self.cfg.weight_file = weight_file
        self.cfg.freeze_embed = freeze_embed
        self.cfg.flat_features = self.cfg.flat_features if flat_features is None else flat_features

        self.backbone, ckpt = init_backbone(self.cfg, self.cfg.weight_file)
        dim = self.backbone.pos_embed.size(-1)

        if num_classes is not None and 'module.head.mlp.mlp.0.weight' in ckpt and \
                ckpt['module.head.mlp.mlp.0.weight'].shape[-1] == dim:
            self.cfg.flat_features = True

        n = 1 if self.cfg.flat_features else (self.cfg.input_size[0] // self.cfg.patch_size[0])
        self.cfg.feature_d = dim * n

        if num_classes is not None:
            self.head_norm = torch.nn.BatchNorm1d(self.cfg.feature_d, affine=False)
            self.head = torch.nn.Linear(self.cfg.feature_d, num_classes)
            from timm.models.layers import trunc_normal_
            trunc_normal_(self.head.weight, std=2e-5)
            load_head_weights(ckpt, self.head_norm, self.head)

        if self.cfg.freeze_embed:
            import models_mae
            models_mae.set_requires_grad(self.backbone.patch_embed, False)

        self.to_spec = create_mel_transform(self.cfg)

    def to_logmel(self, audio):
        mel = self.to_spec(audio)
        return torch.log(mel + torch.finfo().eps).unsqueeze(1)

    def normalize(self, x):
        return (x - self.cfg.mean) / self.cfg.std

    def encode_features(self, x, avg=False):
        p_f = self.backbone.grid_size()[0]
        u_f = self.cfg.input_size[1]
        p_t = self.backbone.patch_size()[1]
        emb_dim = self.backbone.patch_embed.proj.out_channels
        n = (x.size(-1) + u_f - 1) // u_f
        pad = (p_t - (x.size(-1) % u_f % p_t)) % p_t
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad))
        feats = []
        for i in range(n):
            emb = self.backbone.forward_encoder(x[..., i*u_f:(i+1)*u_f])
            emb = emb[..., 1:, :]
            if self.cfg.flat_features:
                if avg:
                    emb = rearrange(emb, 'b (f t) d -> b t d f', f=p_f, d=emb_dim).mean(-1)
                feats.append(emb)
            else:
                emb = rearrange(emb, 'b (f t) d -> b t (f d)', f=p_f, d=emb_dim)
                feats.append(emb)
        return torch.cat(feats, dim=-2)

    def encode(self, batch_audio, avg=False):
        x = self.normalize(self.to_logmel(batch_audio))
        return self.encode_features(x, avg)

    def forward(self, batch_audio, average_per_time_frame=False):
        x = self.encode(batch_audio, average_per_time_frame)
        if hasattr(self, 'head'):
            x = x.mean(1)
            x = self.head_norm(x.unsqueeze(-1)).squeeze(-1)
            return self.head(x)
        return x

    def get_scene_embeddings(self, batch_audio):
        return self.encode(batch_audio).mean(1)

    def get_timestamp_embeddings(self, batch_audio):
        x = self.encode(batch_audio, avg=True)
        ts = compute_timestamps(self.cfg, batch_audio, x)
        return x, ts

    def forward_frames(self, batch_audio):
        x, ts = self.get_timestamp_embeddings(batch_audio)
        if hasattr(self, 'head'):
            x = self.head_norm(x.transpose(-1, -2)).transpose(-2, -1)
            x = self.head(x)
        return x, ts

    def encode_clap_audio(self, batch_audio):
        x = self.forward(batch_audio).mean(dim=-2)
        return self.backbone.audio_proj(x)

    def encode_clap_text(self, texts, truncate=False):
        if not hasattr(self, 'text_encoder'):
            self.text_encoder = GTETextEncoder()
        t = self.text_encoder(texts, truncate=truncate)
        return self.backbone.text_proj(t).detach().cpu().float()

# ============================
# Text Encoder
# ============================
class GTETextEncoder:
    def __init__(self, model_name="thenlper/gte-base"):
        from transformers import AutoTokenizer, AutoModel
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def __call__(self, texts, truncate=True, max_len=512):
        def mean_pool(hidden_states, attn_mask):
            hidden = hidden_states.masked_fill(~attn_mask[..., None].bool(), 0.0)
            return hidden.sum(1) / attn_mask.sum(1)[..., None]
        with torch.no_grad():
            device = next(self.model.parameters()).device
            inputs = self.tokenizer(texts, max_length=max_len, padding=True, truncation=truncate, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
        return mean_pool(outputs.last_hidden_state, inputs['attention_mask'])
