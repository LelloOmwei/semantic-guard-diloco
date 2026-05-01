import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Simple model + dataset
# -----------------------------
def make_data(n=1000):
    X = torch.randn(n, 2)
    y = (X[:, 0] + X[:, 1] > 0).long()
    return X, y

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc(x)

# -----------------------------
# Semantic Encoder φ (linear projection)
# -----------------------------
def semantic_encode(grad):
    # Flatten all gradients into one vector
    flat = torch.cat([g.flatten() for g in grad])
    
    # Deterministic projection to 32D for consistent demo results
    torch.manual_seed(42) 
    proj = torch.randn(flat.shape[0], 32)
    
    atom = flat @ proj
    atom = atom / (atom.norm() + 1e-6)
    return atom

# -----------------------------
# Semantic Guard
# -----------------------------
class SemanticGuard:
    def __init__(self, tau=0.5):
        self.tau = tau

    def score(self, atom, global_atom):
        return torch.dot(atom, global_atom)

    def weight(self, scs):
        # Soft-gating implementation
        return max(0.0, (scs - self.tau) / (1 - self.tau))

# -----------------------------
# Training logic
# -----------------------------
def compute_gradient(model, X, y):
    model.zero_grad()
    out = model(X)
    loss = nn.CrossEntropyLoss()(out, y)
    loss.backward()
    return [p.grad.clone() for p in model.parameters()]

def apply_update(model, grad, lr=0.1):
    with torch.no_grad():
        for p, g in zip(model.parameters(), grad):
            p -= lr * g

# -----------------------------
# Simulation: Baseline vs Semantic Guard
# -----------------------------
def run_experiment(num_nodes=10, malicious_ratio=0.4, epochs=30):
    print(f"Starting simulation with {num_nodes} nodes ({int(malicious_ratio*100)}% malicious)...")
    
    X, y = make_data()
    model_base = SimpleModel()
    model_guard = SimpleModel()

    guard = SemanticGuard(tau=0.3)
    acc_base = []
    acc_guard = []

    global_atom = torch.zeros(32)

    for epoch in range(epochs):
        grads_batch = []
        atoms_batch = []

        for i in range(num_nodes):
            idx = torch.randint(0, len(X), (64,))
            X_i, y_i = X[idx], y[idx]

            # Compute legitimate gradient
            g = compute_gradient(model_base, X_i, y_i)

            # Simulate malicious node (Model Poisoning)
            is_malicious = i < int(num_nodes * malicious_ratio)
            if is_malicious:
                # Malicious nodes send high-magnitude noise
                g = [torch.randn_like(x) * 10.0 for x in g]

            grads_batch.append(g)
            atom = semantic_encode(g)
            atoms_batch.append(atom)

        # Update global semantic context (EMA)
        legit_atoms = torch.stack(atoms_batch)
        global_atom = 0.9 * global_atom + 0.1 * legit_atoms.mean(dim=0)
        global_atom = global_atom / (global_atom.norm() + 1e-6)

        # 1. Baseline Aggregation (Simple Average)
        avg_grad = []
        for params in zip(*grads_batch):
            avg_grad.append(torch.stack(params).mean(dim=0))
        apply_update(model_base, avg_grad)

        # 2. Semantic Guard Aggregation
        weighted_grads = []
        for g, atom in zip(grads_batch, atoms_batch):
            scs = guard.score(atom, global_atom)
            w = guard.weight(scs)
            weighted_grads.append([w * layer_grad for layer_grad in g])

        agg_grad = []
        for params in zip(*weighted_grads):
            agg_grad.append(torch.stack(params).mean(dim=0))
        apply_update(model_guard, agg_grad)

        # Evaluation
        def get_acc(model):
            with torch.no_grad():
                pred = model(X).argmax(dim=1)
                return (pred == y).float().mean().item()

        acc_base.append(get_acc(model_base))
        acc_guard.append(get_acc(model_guard))
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Baseline Acc={acc_base[-1]:.3f}, Guard Acc={acc_guard[-1]:.3f}")

    return acc_base, acc_guard

if __name__ == "__main__":
    base, guard = run_experiment()

    plt.figure(figsize=(10, 6))
    plt.plot(base, label="Baseline (Mean Aggregation)", color='red', linestyle='--')
    plt.plot(guard, label="Semantic Guard (Our Approach)", color='green', linewidth=2)
    plt.axhline(y=0.9, color='gray', linestyle=':', label="Target Accuracy")
    plt.xlabel("Training Epochs")
    plt.ylabel("Model Accuracy")
    plt.title("Poisoning Resistance: Semantic Guarding vs Standard DiLoCo")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    import os
    if not os.path.exists('assets'):
        os.makedirs('assets')
    plt.savefig('assets/plot.png')
    print("Simulation complete. Result saved to assets/plot.png")
