#!/usr/bin/env python3
"""
SpikingBetterAttentionLM: Hybrid BetterAttention + MoE Sin/Cos Spiking Network
=============================================================================

Architecture:
  1. Hybrid BetterAttention:
     - Local heads: Exact sliding-window attention over W past tokens.
     - Global heads: Holographic cognitive map with content-addressable lenses
       mapping pre-RoPE features to a low-dim space (shelf-based routing).
  2. MoE Sin/Cos Spiking Layer (Dimension-Reduction for SNN):
     - Gating Router: Selects top-k experts per token with load-balancing loss.
     - Experts: Compress d_model -> d_expert (low-dim subspace) where resonate-and-fire
       phase dynamics (theta = h @ Omega^T + phi, cos, sin, spikes) avoid phase
       dispersion, then project back up to d_model.

Usage:
  # Train model (runs coherence check every 1000 steps)
  python spiking_better_attention.py --train --epochs 3 --batch_size 8

  # Interactive Chat / Inference (loads saved checkpoint)
  python spiking_better_attention.py --chat
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    def __init__(self, **kw):
        # Model architecture
        self.n_embd = kw.get("n_embd", 256)
        self.n_layer = kw.get("n_layer", 4)
        self.n_head = kw.get("n_head", 8)
        self.n_local_heads = kw.get("n_local_heads", 6)
        self.head_dim = self.n_embd // self.n_head
        self.window = kw.get("window", 256)               # Local sliding-window size
        self.map_dim = kw.get("map_dim", 32)              # Holographic map dimension
        self.top_k_tok = kw.get("top_k_tok", 32)          # Holographic top-k tokens (training)
        self.seg = kw.get("seg", 64)                      # Segment ("shelf") size for decode
        self.top_seg = kw.get("top_seg", 4)               # Shelves skimmed per step
        self.block_size = kw.get("block_size", 512)
        self.dropout = kw.get("dropout", 0.0)
        self.rope_scale = kw.get("rope_scale", 1.0)

        # MoE & SNN Knobs (Lower-dim experts solve SNN phase dispersion)
        self.num_experts = kw.get("num_experts", 4)       # Total spiking experts
        self.top_k_experts = kw.get("top_k_experts", 2)   # Active experts per token
        self.d_expert = kw.get("d_expert", 64)            # Low-dimensional manifold per expert
        self.oscillators = kw.get("oscillators", 128)     # Resonant oscillators per expert
        self.spike_threshold = kw.get("spike_threshold", 0.45)
        self.aux_loss_coef = kw.get("aux_loss_coef", 0.01)

        # Training & IO
        self.tokenizer = kw.get("tokenizer", "tiktoken")  # "tiktoken" or "byte"
        self.batch_size = kw.get("batch_size", 8)
        self.lr = kw.get("lr", 3e-4)
        self.min_lr = kw.get("min_lr", 3e-5)
        self.warmup = kw.get("warmup", 200)
        self.grad_clip = kw.get("grad_clip", 1.0)
        self.epochs = kw.get("epochs", 3)
        self.sample_every = kw.get("sample_every", 1000)
        self.sample_len = kw.get("sample_len", 120)
        self.max_ctx = kw.get("max_ctx", 8192)
        self.input_file = kw.get("input_file", "input.txt")
        self.ckpt = kw.get("ckpt", os.path.join("ckpt", "spiking_better_attention.pt"))

    def to_dict(self):
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# Tokenizer Wrapper (tiktoken cl50k / gpt2 or Byte-Level fallback)
# ---------------------------------------------------------------------------

class Tokenizer:
    def __init__(self, kind: str = "tiktoken"):
        self.kind = kind
        if kind == "tiktoken":
            try:
                import tiktoken
                try:
                    self.enc = tiktoken.get_encoding("cl50k_base")
                except Exception:
                    self.enc = tiktoken.get_encoding("gpt2")
                self.vocab_size = self.enc.n_vocab
            except ImportError:
                print("[!] tiktoken not installed. Falling back to byte-level tokenizer.")
                self.kind = "byte"
                self.vocab_size = 256
        else:
            self.kind = "byte"
            self.vocab_size = 256

    def encode(self, text: str) -> list[int]:
        if self.kind == "tiktoken":
            return self.enc.encode(text)
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        if self.kind == "tiktoken":
            return self.enc.decode(tokens)
        return bytes(tokens).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# RoPE (Rotary Position Embeddings)
# ---------------------------------------------------------------------------

def rope_cos_sin(max_len: int, head_dim: int, device: torch.device, dtype: torch.dtype, base: float = 10000.0, scale: float = 1.0):
    if scale and scale > 1.0:
        base = base * (scale ** (head_dim / (head_dim - 2)))
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2], out[..., 1::2] = o1, o2
    return out


# ---------------------------------------------------------------------------
# 1. Low-Dimensional Sin/Cos Spiking Expert & MoE Layer
# ---------------------------------------------------------------------------

class SinCosSpikingExpert(nn.Module):
    """
    Subspace Resonate-and-Fire Expert:
    Projects d_model -> d_expert to compute harmonic phase interference inside
    a compact manifold, avoiding high-dimensional dispersion.
    """
    def __init__(self, d_model: int, d_expert: int, num_oscillators: int, threshold: float = 0.45):
        super().__init__()
        self.d_model = d_model
        self.d_expert = d_expert
        self.threshold = threshold

        # Subspace compression
        self.in_proj = nn.Linear(d_model, d_expert, bias=False)

        # Spectral scaling prevents phase variance explosion inside the subspace
        freq_scale = 1.0 / math.sqrt(d_expert)
        self.omega = nn.Parameter(torch.randn(num_oscillators, d_expert) * freq_scale)
        self.phi = nn.Parameter(torch.zeros(num_oscillators))

        # Nonlinear synaptic readout mapping [cos, sin, spikes] back to d_model
        self.out_proj = nn.Sequential(
            nn.Linear(num_oscillators * 3, d_expert * 2),
            nn.GELU(),
            nn.Linear(d_expert * 2, d_model, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, d_model]
        Returns: [N, d_model]
        """
        h = self.in_proj(x)  # Compressed manifold: [N, d_expert]
        theta = F.linear(h, self.omega, self.phi)  # Instantaneous phase: [N, num_oscillators]

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        spikes = F.relu(cos_theta - self.threshold)  # Resonate-and-fire action potentials

        harmonic_features = torch.cat([cos_theta, sin_theta, spikes], dim=-1)
        return self.out_proj(harmonic_features)


class MoESpikingLayer(nn.Module):
    """
    Top-k Gated Sparse Mixture of Sin/Cos Spiking Experts with load balancing.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.d_model = cfg.n_embd
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k_experts
        self.router = nn.Linear(cfg.n_embd, cfg.num_experts, bias=False)

        self.experts = nn.ModuleList([
            SinCosSpikingExpert(
                d_model=cfg.n_embd,
                d_expert=cfg.d_expert,
                num_oscillators=cfg.oscillators,
                threshold=cfg.spike_threshold
            )
            for _ in range(cfg.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        x_flat = x.view(-1, self.d_model)  # [N, d_model]
        N = x_flat.size(0)

        router_logits = self.router(x_flat)  # [N, E]
        routing_probs = F.softmax(router_logits, dim=-1)
        top_weights, top_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)  # Normalize weights

        # Compute Switch/GShard auxiliary load-balancing loss: E * sum(f_e * P_e)
        if self.training:
            f_e = torch.zeros(self.num_experts, device=x.device)
            f_e.scatter_add_(0, top_indices.reshape(-1), torch.full_like(top_indices.reshape(-1), 1.0 / (N * self.top_k), dtype=torch.float32))
            P_e = routing_probs.mean(dim=0)
            aux_loss = self.num_experts * torch.sum(f_e * P_e)
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        # Dispatch tokens to experts via index_add
        out = torch.zeros_like(x_flat)
        for e_idx, expert in enumerate(self.experts):
            mask = (top_indices == e_idx)
            if not mask.any():
                continue
            token_idx, k_idx = torch.where(mask)
            expert_in = x_flat[token_idx]
            expert_out = expert(expert_in)
            weights = top_weights[token_idx, k_idx].unsqueeze(-1)
            out.index_add_(0, token_idx, expert_out * weights)

        return out.view(orig_shape), aux_loss


# ---------------------------------------------------------------------------
# 2. Hybrid Attention Layer (Sliding Window + Holographic Cognitive Map)
# ---------------------------------------------------------------------------

class HybridAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        H, HL, hd, m = cfg.n_head, cfg.n_local_heads, cfg.head_dim, cfg.map_dim
        self.H, self.HL, self.HG, self.hd, self.m = H, HL, H - HL, hd, m
        assert self.HG > 0, "Requires at least one holographic (global) head"

        self.q = nn.Linear(cfg.n_embd, H * hd, bias=False)
        self.k = nn.Linear(cfg.n_embd, H * hd, bias=False)
        self.v = nn.Linear(cfg.n_embd, H * hd, bias=False)
        self.out = nn.Linear(H * hd, cfg.n_embd, bias=False)

        # Holographic content-addressable lenses (operate pre-RoPE)
        self.lens_q = nn.Linear(hd * self.HG, m, bias=False)
        self.lens_k = nn.Linear(hd * self.HG, m, bias=False)
        for lin in (self.lens_q, self.lens_k):
            nn.init.normal_(lin.weight, std=0.02)

    def forward_train(self, x: torch.Tensor, causal: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        cfg = self.cfg
        q = self.q(x).view(B, T, self.H, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.H, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.H, self.hd).transpose(1, 2)

        q_content, k_content = q, k
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Local heads: exact sliding window attention
        t_idx = torch.arange(T, device=x.device)
        win = (t_idx[None, :] - t_idx[:, None]) < cfg.window
        mask = (causal & win)[None, None]
        out_l = F.scaled_dot_product_attention(
            q[:, :self.HL], k[:, :self.HL], v[:, :self.HL], attn_mask=mask
        )

        # Global heads: holographic top-k retrieval
        q_g, k_g, v_g = q[:, self.HL:], k[:, self.HL:], v[:, self.HL:]
        pq = self.lens_q(q_content[:, self.HL:].permute(0, 2, 1, 3).reshape(B, T, -1))
        pk = self.lens_k(k_content[:, self.HL:].permute(0, 2, 1, 3).reshape(B, T, -1))

        scores = torch.einsum("btm,bsm->bts", pq, pk) / math.sqrt(self.m)
        scores = scores.masked_fill(~causal, float("-inf"))
        K = min(cfg.top_k_tok, T)
        topv, topi = scores.topk(K, dim=-1)

        topi_h = topi[:, None].expand(-1, self.HG, -1, -1)
        flat_idx = topi_h.reshape(B * self.HG, T * K, 1).expand(-1, -1, self.hd)
        k_sel = k_g.reshape(B * self.HG, T, self.hd).gather(1, flat_idx).view(B, self.HG, T, K, self.hd)
        v_sel = v_g.reshape(B * self.HG, T, self.hd).gather(1, flat_idx).view(B, self.HG, T, K, self.hd)

        logits = torch.einsum("bhtd,bhtkd->bhtk", q_g, k_sel) / math.sqrt(self.hd)
        logits = logits + topv[:, None].detach()
        attn = F.softmax(logits, dim=-1)
        out_g = torch.einsum("bhtk,bhtkd->bhtd", attn, v_sel)

        out = torch.cat([out_l, out_g], dim=1).transpose(1, 2).reshape(B, T, -1)
        return self.out(out)

    @staticmethod
    def init_cache(cap: int, device: torch.device, dtype: torch.dtype):
        return {"k": None, "v": None, "pk": None, "n": 0, "cap": cap, "seg_means": None, "frozen": 0}

    def step(self, x_t: torch.Tensor, cache: dict, cos1: torch.Tensor, sin1: torch.Tensor, pos: int):
        cfg = self.cfg
        q = self.q(x_t).view(1, self.H, 1, self.hd)
        k = self.k(x_t).view(1, self.H, 1, self.hd)
        v = self.v(x_t).view(1, self.H, 1, self.hd)
        hg_content = q[:, self.HL:]
        pq = self.lens_q(hg_content.permute(0, 2, 1, 3).reshape(1, -1))

        if cache["k"] is None:
            dt = k.dtype
            cache["k"] = torch.zeros(1, self.H, cache["cap"], self.hd, device=k.device, dtype=dt)
            cache["v"] = torch.zeros(1, self.H, cache["cap"], self.hd, device=k.device, dtype=dt)
            cache["pk"] = torch.zeros(1, self.HG, cache["cap"], self.m, device=k.device, dtype=dt)

        n = cache["n"]
        cache["k"][:, :, n, :] = k[:, :, 0, :]
        cache["v"][:, :, n, :] = v[:, :, 0, :]
        cache["pk"][:, :, n, :] = pq
        cache["n"] = n + 1
        n += 1
        kc, vc, pkc = cache["k"], cache["v"], cache["pk"]

        q = apply_rope(q, cos1[pos].view(1, 1, 1, -1), sin1[pos].view(1, 1, 1, -1))

        # Local heads: exact attention over window W
        start = max(0, n - cfg.window)
        k_l = apply_rope(kc[:, :self.HL, start:n, :], cos1[start:n].view(1, 1, n - start, -1), sin1[start:n].view(1, 1, n - start, -1)).to(q.dtype)
        v_l = vc[:, :self.HL, start:n, :].to(q.dtype)
        out_l = F.scaled_dot_product_attention(q[:, :self.HL], k_l, v_l)

        # Global heads: shelf routing
        frozen = n // cfg.seg
        if cache["seg_means"] is None:
            cache["seg_means"] = torch.zeros(1, self.HG, 0, self.m, device=kc.device, dtype=kc.dtype)
        while cache["frozen"] < frozen:
            s = cache["frozen"]
            seg_mean = pkc[:, :, s * cfg.seg:(s + 1) * cfg.seg, :].mean(dim=2, keepdim=True)
            cache["seg_means"] = torch.cat([cache["seg_means"], seg_mean], dim=2)
            cache["frozen"] += 1

        seg_means = cache["seg_means"][0].to(pq.dtype)
        if seg_means.size(1) > 0:
            seg_scores = torch.einsum("hm,hsm->hs", pq.expand(self.HG, -1), seg_means) / math.sqrt(self.m)
        else:
            seg_scores = pq.new_zeros(self.HG, 0)

        part = pkc[:, :, frozen * cfg.seg:n, :][0].to(pq.dtype)
        part_scores = torch.einsum("hm,hpm->hp", pq.expand(self.HG, -1), part) / math.sqrt(self.m)
        all_scores = torch.cat([seg_scores, part_scores], dim=1)
        n_sel = min(cfg.top_seg, all_scores.size(1))
        top = all_scores.topk(n_sel, dim=1).indices

        pos_lists = []
        for h in range(self.HG):
            toks = []
            for j in top[h].tolist():
                if j < frozen:
                    toks.extend(range(j * cfg.seg, (j + 1) * cfg.seg))
                else:
                    toks.extend(range(frozen * cfg.seg, n))
            pos_lists.append(toks)

        L = max(len(p) for p in pos_lists)
        tok_idx = torch.full((self.HG, L), n - 1, dtype=torch.long, device=kc.device)
        pad_mask = torch.zeros(self.HG, L, dtype=torch.bool, device=kc.device)
        for h, p in enumerate(pos_lists):
            if p:
                tok_idx[h, :len(p)] = torch.tensor(p, device=kc.device)
                pad_mask[h, :len(p)] = True

        g_idx = tok_idx[:, :, None].expand(1, -1, -1, self.hd)
        k_sel = kc[:, self.HL:].gather(2, g_idx)
        k_sel = apply_rope(k_sel, cos1[tok_idx].unsqueeze(0), sin1[tok_idx].unsqueeze(0)).to(q.dtype)
        v_sel = vc[:, self.HL:].gather(2, g_idx).to(q.dtype)
        logits = (q[:, self.HL:] * k_sel).sum(-1) / math.sqrt(self.hd)
        logits = logits.masked_fill(~pad_mask[None], float("-inf"))
        attn = F.softmax(logits, dim=-1)
        out_g = (attn.unsqueeze(-1) * v_sel).sum(2, keepdim=True)

        out = torch.cat([out_l, out_g], dim=1).transpose(1, 2).reshape(1, 1, -1)
        return self.out(out), cache

    @torch.no_grad()
    def prefill_chunk(self, x: torch.Tensor, cache: dict, pos0: int, cos1: torch.Tensor, sin1: torch.Tensor):
        B, T, C = x.shape
        cfg = self.cfg
        q = self.q(x).view(1, T, self.H, self.hd).transpose(1, 2)
        k = self.k(x).view(1, T, self.H, self.hd).transpose(1, 2)
        v = self.v(x).view(1, T, self.H, self.hd).transpose(1, 2)
        pq = self.lens_q(q[:, self.HL:].permute(0, 2, 1, 3).reshape(T, -1))

        if cache["k"] is None:
            dt = k.dtype
            cache["k"] = torch.zeros(1, self.H, cache["cap"], self.hd, device=k.device, dtype=dt)
            cache["v"] = torch.zeros(1, self.H, cache["cap"], self.hd, device=k.device, dtype=dt)
            cache["pk"] = torch.zeros(1, self.HG, cache["cap"], self.m, device=k.device, dtype=dt)

        n0 = cache["n"]
        n = n0 + T
        cache["k"][:, :, n0:n, :] = k
        cache["v"][:, :, n0:n, :] = v
        cache["pk"][:, :, n0:n, :] = pq
        cache["n"] = n
        kc, vc, pkc = cache["k"], cache["v"], cache["pk"]

        q_rope = apply_rope(q, cos1[pos0:pos0 + T].view(1, 1, T, -1), sin1[pos0:pos0 + T].view(1, 1, T, -1))

        # Local heads
        k_all = apply_rope(kc[:, :self.HL, :n, :], cos1[:n].view(1, 1, n, -1), sin1[:n].view(1, 1, n, -1)).to(q_rope.dtype)
        v_all = vc[:, :self.HL, :n, :].to(q_rope.dtype)
        lmask = torch.arange(n, device=x.device)[None, :] <= torch.arange(n0, n, device=x.device)[:, None]
        out_l = F.scaled_dot_product_attention(q_rope[:, :self.HL], k_all, v_all, attn_mask=lmask[None, None])

        # Global heads
        frozen = n // cfg.seg
        if cache["seg_means"] is None:
            cache["seg_means"] = torch.zeros(1, self.HG, 0, self.m, device=x.device, dtype=k.dtype)
        while cache["frozen"] < frozen:
            s = cache["frozen"]
            seg_mean = pkc[:, :, s * cfg.seg:(s + 1) * cfg.seg, :].mean(dim=2, keepdim=True)
            cache["seg_means"] = torch.cat([cache["seg_means"], seg_mean], dim=2)
            cache["frozen"] += 1

        sm = cache["seg_means"][0, 0]
        P = n - frozen * cfg.seg
        S = frozen
        js = torch.arange(S + P, device=x.device)
        starts = torch.where(js < S, js * cfg.seg, S * cfg.seg + (js - S))
        lens = torch.where(js < S, torch.full_like(js, cfg.seg), torch.ones_like(js))
        Lm = int(lens.max().item())
        tbl = (starts[:, None] + torch.arange(Lm, device=x.device)[None, :]).clamp(max=n - 1)
        tvalid = torch.arange(Lm, device=x.device)[None, :] < lens[:, None]

        parts = [(pq @ sm.to(pq.dtype).T) / math.sqrt(self.m)]
        if P > 0:
            part = pkc[0, 0, S * cfg.seg:n, :].to(pq.dtype)
            parts.append((pq @ part.T) / math.sqrt(self.m))
        all_scores = torch.cat(parts, dim=-1)
        qpos = torch.arange(n0, n, device=x.device)
        allow = starts[None, :] <= qpos[:, None]
        all_scores = all_scores.masked_fill(~allow, float("-inf"))
        ns = min(cfg.top_seg, S + P)
        top = all_scores.topk(ns, dim=-1).indices

        sel_tbl = tbl[top]
        sel_val = tvalid[top] & (sel_tbl <= qpos[:, None, None])
        K = ns * Lm
        tok_idx = sel_tbl.reshape(T, K)
        val = sel_val.reshape(T, K)

        tok_h = tok_idx[None].expand(self.HG, -1, -1)
        g_idx = tok_h[:, :, :, None].expand(self.HG, T, K, self.hd)
        k_sel = kc[0, self.HL:].gather(1, g_idx.reshape(self.HG, T * K, self.hd)).reshape(self.HG, T, K, self.hd)
        k_sel = apply_rope(k_sel, cos1[tok_h], sin1[tok_h])
        v_sel = vc[0, self.HL:].gather(1, g_idx.reshape(self.HG, T * K, self.hd)).reshape(self.HG, T, K, self.hd)
        logits = (q_rope[0, self.HL:].unsqueeze(2) * k_sel).sum(-1) / math.sqrt(self.hd)
        logits = logits.masked_fill(~val, float("-inf"))
        attn = F.softmax(logits, dim=-1)
        out_g = (attn.unsqueeze(-1) * v_sel).sum(2)

        out = torch.cat([out_l, out_g.unsqueeze(0)], dim=1).transpose(1, 2).reshape(1, T, -1)
        return self.out(out), cache


# ---------------------------------------------------------------------------
# 3. Hybrid Block & Complete LM
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = HybridAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.moe = MoESpikingLayer(cfg)

    def forward_train(self, x: torch.Tensor, causal: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn.forward_train(self.ln1(x), causal, cos, sin)
        moe_out, aux_loss = self.moe(self.ln2(x))
        x = x + moe_out
        return x, aux_loss

    def step(self, x_t: torch.Tensor, cache: dict, cos1: torch.Tensor, sin1: torch.Tensor, pos: int):
        a, cache = self.attn.step(self.ln1(x_t), cache, cos1, sin1, pos)
        x_t = x_t + a
        moe_out, _ = self.moe(self.ln2(x_t))
        x_t = x_t + moe_out
        return x_t, cache

    def prefill_chunk(self, x: torch.Tensor, cache: dict, pos0: int, cos1: torch.Tensor, sin1: torch.Tensor):
        a, cache = self.attn.prefill_chunk(self.ln1(x), cache, pos0, cos1, sin1)
        x = x + a
        moe_out, _ = self.moe(self.ln2(x))
        x = x + moe_out
        return x, cache


class SpikingBetterAttentionLM(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # Weight tying

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward_train(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = idx.shape
        cfg = self.cfg
        x = self.tok_emb(idx)
        causal = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
        cos, sin = rope_cos_sin(T, cfg.head_dim, idx.device, x.dtype, scale=cfg.rope_scale)

        total_aux_loss = torch.tensor(0.0, device=idx.device)
        for blk in self.blocks:
            x, aux_loss = blk.forward_train(x, causal, cos, sin)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, total_aux_loss

    def init_caches(self, cap: int, device: torch.device, dtype: torch.dtype):
        return [self.blocks[i].attn.init_cache(cap, device, dtype) for i in range(self.cfg.n_layer)]

    def step(self, tok: torch.Tensor, caches: list, pos: int, cos1: torch.Tensor, sin1: torch.Tensor):
        x = self.tok_emb(tok).view(1, 1, -1)
        for i, blk in enumerate(self.blocks):
            x, caches[i] = blk.step(x, caches[i], cos1, sin1, pos)
        return self.head(self.ln_f(x))[:, -1, :], caches

    @torch.no_grad()
    def prefill_chunk(self, toks: torch.Tensor, caches: list, pos0: int, cos1: torch.Tensor, sin1: torch.Tensor, chunk: int = 512):
        logits = None
        for i in range(0, len(toks), chunk):
            c = toks[i:i + chunk]
            x = self.tok_emb(c).unsqueeze(0)
            for li, blk in enumerate(self.blocks):
                x, _ = blk.prefill_chunk(x, caches[li], pos0 + int(i), cos1, sin1)
            x = self.head(self.ln_f(x))
            logits = x[:, -1, :].float()
        return logits


# ---------------------------------------------------------------------------
# Sampling and Generation
# ---------------------------------------------------------------------------

def sample_token(logits: torch.Tensor, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9, rep_penalty: float = 1.1, rep_tokens: list = None) -> int:
    logits = logits[0].float()
    if rep_penalty != 1.0 and rep_tokens:
        for t in list(rep_tokens)[-192:]:
            t = int(t)
            if logits[t] > 0:
                logits[t] /= rep_penalty
            else:
                logits[t] *= rep_penalty
    logits = logits / max(temperature, 1e-5)
    if top_k and top_k > 0:
        vk, ik = logits.topk(min(top_k, logits.size(-1)))
        probs = torch.zeros_like(logits).scatter_(0, ik, F.softmax(vk, dim=-1))
    else:
        probs = F.softmax(logits, dim=-1)
    if top_p and top_p < 1.0:
        sp, si = probs.sort(descending=True)
        cum = sp.cumsum(-1)
        cut = cum > top_p
        cut[1:] = cut[:-1].clone()
        cut[0] = False
        probs[si[cut]] = 0.0
        probs = probs / probs.sum()
    return torch.multinomial(probs, 1).item()


@torch.no_grad()
def generate_sample(model: SpikingBetterAttentionLM, tokenizer: Tokenizer, prompt: str, length: int = 100, device: torch.device = "cpu", temperature: float = 0.8) -> str:
    was_training = model.training
    model.eval()
    tokens = tokenizer.encode(prompt)
    total_len = len(tokens) + length + 8
    caches = model.init_caches(total_len, device, torch.float32)
    cos, sin = rope_cos_sin(total_len, model.cfg.head_dim, device, torch.float32)

    toks_t = torch.tensor(tokens, dtype=torch.long, device=device)
    logits = model.prefill_chunk(toks_t, caches, 0, cos, sin) if len(tokens) > 0 else None

    out = []
    for i in range(length):
        nt = sample_token(logits, temperature=temperature, top_k=50, top_p=0.9, rep_penalty=1.1, rep_tokens=tokens + out)
        out.append(nt)
        logits, caches = model.step(torch.tensor([nt], device=device), caches, len(tokens) + i, cos, sin)

    if was_training:
        model.train()
    return tokenizer.decode(tokens + out)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(model: SpikingBetterAttentionLM, tokenizer: Tokenizer, cfg: Config, data_tensor: torch.Tensor, device: torch.device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = max(1, (len(data_tensor) - cfg.block_size) // (cfg.batch_size * cfg.block_size))
    total_steps = steps_per_epoch * cfg.epochs

    def get_batch():
        max_idx = len(data_tensor) - cfg.block_size - 1
        starts = torch.randint(0, max_idx, (cfg.batch_size,))
        x = torch.stack([data_tensor[i:i + cfg.block_size] for i in starts]).to(device)
        y = torch.stack([data_tensor[i + 1:i + cfg.block_size + 1] for i in starts]).to(device)
        return x, y

    def lr_at(step):
        if step < cfg.warmup:
            return cfg.lr * (step + 1) / cfg.warmup
        p = (step - cfg.warmup) / max(1, total_steps - cfg.warmup)
        return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * p))

    model.train()
    t0 = time.time()
    global_step = 0
    print(f"[*] Training SpikingBetterAttentionLM: {total_steps:,} steps across {cfg.epochs} epochs")

    for epoch in range(1, cfg.epochs + 1):
        for epoch_step in range(steps_per_epoch):
            global_step += 1
            lr = lr_at(global_step)
            for g in opt.param_groups:
                g["lr"] = lr

            x, y = get_batch()
            logits, aux_loss = model.forward_train(x)
            ce_loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
            total_loss = ce_loss + cfg.aux_loss_coef * aux_loss

            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            if global_step % 50 == 0 or global_step == 1:
                elapsed = time.time() - t0
                tok_s = (global_step * cfg.batch_size * cfg.block_size) / max(1e-5, elapsed)
                print(f"[Epoch {epoch}/{cfg.epochs} | Step {global_step:>5}/{total_steps}] Loss: {total_loss.item():.4f} (CE: {ce_loss.item():.4f}, Aux: {aux_loss.item():.4f}) | {tok_s:.0f} tok/s")

            # Coherence check
            if global_step % cfg.sample_every == 0 or global_step == 1:
                print("\n" + "=" * 65)
                print(f"  >>> COHERENCE CHECK @ STEP {global_step} <<<")
                prompt = "He was going to"
                sample = generate_sample(model, tokenizer, prompt=prompt, length=cfg.sample_len, device=device)
                print(f"SAMPLE:\n{sample.strip()}")
                print("=" * 65 + "\n")

    os.makedirs(os.path.dirname(cfg.ckpt) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "vocab_size": model.vocab_size}, cfg.ckpt)
    print(f"[✓] Training complete. Saved model to {cfg.ckpt}")


# ---------------------------------------------------------------------------
# Interactive Chat Mode
# ---------------------------------------------------------------------------

@torch.no_grad()
def chat_mode(model: SpikingBetterAttentionLM, tokenizer: Tokenizer, cfg: Config, device: torch.device):
    model.eval()
    max_ctx = cfg.max_ctx
    caches = model.init_caches(max_ctx + 64, device, torch.float32)
    pos = 0
    cos, sin = rope_cos_sin(max_ctx + 64, cfg.head_dim, device, torch.float32, scale=cfg.rope_scale)
    print("\n[Chat] SpikingBetterAttentionLM Ready. Commands: /clear /exit")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/clear":
            caches = model.init_caches(max_ctx + 64, device, torch.float32)
            pos = 0
            print("   [context cleared]")
            continue

        prompt_tokens = tokenizer.encode(user + "\n")
        if pos + len(prompt_tokens) + 100 > max_ctx:
            print("   [context limit hit — resetting]")
            caches = model.init_caches(max_ctx + 64, device, torch.float32)
            pos = 0

        toks_t = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
        logits = model.prefill_chunk(toks_t, caches, pos, cos, sin)
        pos += len(prompt_tokens)

        gen = []
        for _ in range(120):
            nt = sample_token(logits, temperature=0.75, top_k=50, top_p=0.9, rep_penalty=1.1, rep_tokens=prompt_tokens + gen)
            gen.append(nt)
            logits, caches = model.step(torch.tensor([nt], device=device), caches, pos, cos, sin)
            pos += 1
            if nt in [10, tokenizer.encode("\n")[0]] and len(gen) > 40:
                break

        print(f"model> {tokenizer.decode(gen).strip()}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SpikingBetterAttentionLM with MoE")
    parser.add_argument("--train", action="store_true", help="Force training")
    parser.add_argument("--chat", action="store_true", help="Force chat/generation")
    parser.add_argument("--input", default="input.txt", help="Input corpus")
    parser.add_argument("--ckpt", default=os.path.join("ckpt", "spiking_better_attention.pt"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--d_expert", type=int, default=64, help="Subspace dimension per expert")
    parser.add_argument("--oscillators", type=int, default=128, help="Oscillators per expert")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    print(f"[*] Hardware device: {str(device).upper()}")

    cfg = Config(
        input_file=args.input,
        ckpt=args.ckpt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        num_experts=args.num_experts,
        d_expert=args.d_expert,
        oscillators=args.oscillators,
    )

    tokenizer = Tokenizer(cfg.tokenizer)

    # Resume from checkpoint for chat if requested or available
    if os.path.exists(cfg.ckpt) and not args.train:
        print(f"[*] Loading checkpoint from {cfg.ckpt}...")
        sd = torch.load(cfg.ckpt, map_location=device, weights_only=False)
        model = SpikingBetterAttentionLM(cfg, sd["vocab_size"]).to(device)
        model.load_state_dict(sd["model"])
        chat_mode(model, tokenizer, cfg, device)
        return

    # Create corpus if missing
    if not os.path.exists(cfg.input_file):
        print(f"[!] '{cfg.input_file}' not found. Creating starter corpus...")
        sample_corpus = (
            "The brain does not process information via traditional backpropagation.\n"
            "Instead, cortical networks utilize subthreshold membrane oscillations.\n"
            "Neurons behave as resonate-and-fire systems governed by sine and cosine phase dynamics.\n"
            "When waves arrive in phase, constructive interference pushes the membrane potential\n"
            "past the spiking threshold, discharging an action potential in a single cycle.\n"
            "By partitioning high dimensions into a Mixture of Experts, harmonic phase variance\n"
            "remains bounded while holographic cognitive lenses skim context memory efficiently.\n"
        ) * 50
        with open(cfg.input_file, "w", encoding="utf-8") as f:
            f.write(sample_corpus)

    with open(cfg.input_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    if len(tokens) <= cfg.block_size:
        tokens = tokens * ((cfg.block_size * 4) // len(tokens) + 1)
    data_tensor = torch.tensor(tokens, dtype=torch.long)

    model = SpikingBetterAttentionLM(cfg, tokenizer.vocab_size).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Initialized SpikingBetterAttentionLM: {param_count:,} parameters ({cfg.num_experts} experts @ d_expert={cfg.d_expert})")

    train(model, tokenizer, cfg, data_tensor, device)


if __name__ == "__main__":
    main()