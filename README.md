# Semantic Guard for DiLoCo (Lite)

> Stop model poisoning in decentralized training — without inspecting raw gradients.

---

## 💡 The Core Idea: Semantic Gating

Instead of validating gradients numerically, we validate them **semantically**. 
Each gradient is projected into a compact representation:

`g → φ(g) → a ∈ ℝ^32` 

We call this a **Semantic Atom (32BSA)**.

## 🧠 Architecture Overview

This implementation separates decentralized training into two parallel layers:

1. **Numeric Channel (Data Plane)**
   - Raw gradients used for model updates.
   - Compatible with DiLoCo / FedAvg / standard aggregation.

2. **Semantic Channel (Control Plane)**
   - Lightweight validation layer.
   - Operates on semantic atoms instead of raw gradients.

👉 This decoupling allows validation without directly inspecting gradient values.

## 🛠️ How It Works

1. **Semantic Encoding (ϕ)**: `a_i = φ(g_i)` 
2. **Semantic Consistency Score (SCS)**: `scs_i = similarity(a_i, A_global)` 
3. **Trust-Weighted Gating**: `w_i = max(0, (scs_i - τ) / (1 - τ))` 
   *Updates are softly weighted instead of hard-rejected.*
4. **Dual-Track Consensus**: We maintain two global semantic states (Fast & Slow atoms) to ensure both agility and stability.

## 🛡️ Security Intuition

This approach helps detect:
- Semantically inconsistent updates
- Coordinated poisoning attacks
- Distribution shifts

Even when gradients appear numerically plausible, their "semantic intent" must align with the global model's trajectory.

## 📈 Why This Matters

- **Resilience**: Adds a semantic validation layer on top of existing defenses.
- **Efficiency**: Lightweight, runs alongside training.
- **Composable**: Works with existing distributed training setups.
- **Privacy-Ready**: Compatible with Homomorphic Encryption (HE) and SMPC because semantic scoring reduces to dot products.

## ⚖️ License & IP

This repository provides a research and reference implementation. The broader architecture (including hardware-oriented execution and semantic encoding strategies) is part of ongoing proprietary work. 

If you're interested in collaboration or licensing, feel free to reach out.
