# SpikingBetterAttentionLM

A hybrid language model architecture uniting **BetterAttention** (dual-rate sliding-window + holographic memory retrieval) with a **Sparse Mixture-of-Experts (MoE) Sin/Cos Spiking Neural Network**.

---

## 🧠 Architectural Intuition

Standard Transformers suffer from two major computational bottlenecks:
1. **$O(T^2)$ Attention Bottlenecks:** Linearly comparing every token to every past token exhausts compute and expands the KV cache uncontrollably.
2. **Dense Non-Linear Processing:** High-dimensional static feed-forward layers lack sparse, wave-based temporal coordination found in biological cognition.

**SpikingBetterAttentionLM** solves both issues by pairing a human-like dual-memory system with acoustic, biological wave computing.

```
                  ┌──────────────────────────────────────────────┐
                  │                 Input Token                  │
                  └──────────────────────┬───────────────────────┘
                                         ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │ 1. BETTERATTENTION (Dual Cognitive Retrieval)                             │
   │                                                                           │
   │   LOCAL HEADS (75%):                GLOBAL HEADS (25%):                   │
   │   • Exact sliding window (W=256)    • Holographic Content Lenses (dim=32) │
   │   • Immediate syntax & grammar      • "Skims" chapter/shelf summaries     │
   │   • Strict O(W) cache footprint     • Exact attention only on top shelves │
   └─────────────────────────────────────┬─────────────────────────────────────┘
                                         ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │ 2. MoE SIN/COS SPIKING LAYER (Wave Interference in Subspaces)             │
   │                                                                           │
   │                    ┌─────────────────────────────┐                        │
   │                    │     Top-k Gating Router     │                        │
   │                    └──────────────┬──────────────┘                        │
   │                                   │ Dispatches to best 2 experts          │
   │           ┌───────────────────────┴───────────────────────┐               │
   │           ▼                                               ▼               │
   │    ┌──────────────┐                                ┌──────────────┐       │
   │    │   Expert 1   │                                │   Expert 2   │       │
   │    │ (Subspace)   │                                │ (Subspace)   │       │
   │    └──────┬───────┘                                └──────┬───────┘       │
   │           │ Compress: d_model (256) -> d_expert (64)      │               │
   │           ▼                                               ▼               │
   │   Harmonic Resonance (θ = hΩᵀ + φ)                Harmonic Resonance      │
   │   Constructive Wave Interference                  Constructive Waves      │
   │   Discharge Spike: ReLU(cos θ - threshold)        Discharge Spike         │
   │           │                                               │               │
   │           │ Expand: d_expert (64) -> d_model (256)        │               │
   │           └───────────────────────┬───────────────────────┘               │
   │                                   ▼                                       │
   │                     Weighted Expert Combination (+)                       │
   └─────────────────────────────────────┬─────────────────────────────────────┘
                                         ▼
                                   Output Logits
```

---

## 🔬 Core Components

### 1. BetterAttention: The Dual-Memory Library
Instead of treating all attention heads uniformly, heads are split into two complementary systems:

* **Local Heads (Working Memory):** Exact sliding-window attention over the last $W$ tokens ($O(W)$ cost). Handles local grammar, punctuation, and immediate syntactic cohesion.
* **Global Heads (Holographic Index / "Skimming the Shelves"):** 
  * Tokens are organized into discrete memory **shelves** (e.g., 64 tokens).
  * A low-dimensional, content-addressable linear lens projects pre-RoPE semantics into a map space ($m=32$).
  * The query evaluates against **shelf summary vectors** (like skimming book chapters) rather than every past token. Only the highest-scoring shelves are unpacked for exact attention.

---

### 2. Sin/Cos Spiking Neurons: Wave Interference as Computing
Rather than using static activations ($\text{GELU}$ / $\text{ReLU}$), neurons act as **Resonate-and-Fire Harmonic Oscillators**:

* **Carrier Waves:** Input signals modulate an internal phase vector:
  $$\theta = x\Omega^T + \phi$$
  generating real ($\cos\theta$) and imaginary ($\sin\theta$) carrier trajectories.
* **Constructive Interference & Spiking:** When features arrive in phase, wave interference triggers an action potential:
  $$\text{Spike} = \text{ReLU}(\cos\theta - V_{\text{threshold}})$$
* **Harmonic Hilbert Projections:** By Bochner's theorem, this projects inputs into a continuous, phase-coherent coordinate space where complex relational features become linearly separable.

---

### 3. The MoE Breakthrough: Solving the SNN Dimension Bottleneck
**The Problem:** Harmonic resonate-and-fire models fail in large embedding dimensions ($d_{\text{model}} \ge 256$). Due to the geometry of high-dimensional spaces, phase vectors undergo a random walk—waves suffer from **destructive interference and phase dispersion**, causing neurons to misfire or go silent.

**The Solution:** The **Sparse Mixture of Experts (MoE)** acts as an array of acoustic isolation chambers:
1. **Dynamic Routing:** A router dispatches each token to its top-2 specialized experts.
2. **Subspace Compression ($d_{\text{model}} \to d_{\text{expert}}$):** The token is compressed from 256 dimensions down to an isolated 64-dimensional manifold.
3. **Clean Phase Dynamics:** Within this tight subspace, oscillators resonate without phase dilution. Spikes discharge cleanly upon constructive interference.
4. **Synthesis & Readout:** The expert synthesizes carriers and spikes $[\cos\theta, \sin\theta, \text{spikes}]$, projects back up to $d_{\text{model}}$, and recombines outputs via learned gating weights with auxiliary load balancing.

---

## ⚡ Quickstart

### Train
```bash
python spiking_better_attention.py --train --epochs 3 --batch_size 8 --block_size 512
```

### Chat / Inference
```bash
python spiking_better_attention.py --chat
```
```

---
