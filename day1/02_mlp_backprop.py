"""
Example 02 — MLP with Backpropagation from Scratch
====================================================

Goal of this lesson
-------------------
Implement a 2-layer MLP (one hidden layer) trained with backpropagation.
We use the XOR problem as the test case because it is *not* linearly
separable — a single neuron (like day 1) cannot solve it, but adding
one hidden layer can.

Architecture
------------
    Input (2)  →  Hidden (8, ReLU)  →  Output (1, Sigmoid)  →  BCE Loss

What we cover:
  1. Forward pass   : compute predictions, storing intermediate values
  2. Loss           : Binary Cross-Entropy (BCE)
  3. Backward pass  : chain rule through Sigmoid → Linear → ReLU → Linear
  4. Weight update  : same gradient descent as day 1, now for 4 tensors

Run it:
    python day1/02_mlp_backprop.py

Saves a plot to day1/mlp_backprop.png.
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Dataset — XOR (not linearly separable)
# ---------------------------------------------------------------------------
# XOR: output is 1 when x1 and x2 have the same sign, 0 otherwise.
# No straight line can separate the two classes, so the model MUST learn
# a non-linear decision boundary via the hidden layer.

def make_xor_data(n=400, noise=0.15, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(float).reshape(-1, 1)
    X += rng.normal(0, noise, X.shape)   # add noise to make it realistic
    return X, y


# ---------------------------------------------------------------------------
# 2. Activations and their derivatives
# ---------------------------------------------------------------------------

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    """d(relu)/dz — 1 where z > 0, 0 elsewhere."""
    return (z > 0).astype(float)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))   # clip for stability


# ---------------------------------------------------------------------------
# 3. Loss — Binary Cross-Entropy (BCE)
# ---------------------------------------------------------------------------
#
#   L = -(1/n) * Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ]
#
# Lower is better. Returns a scalar.

def bce_loss(y_true, y_pred):
    eps = 1e-9   # prevent log(0)
    return -np.mean(y_true * np.log(y_pred + eps) +
                    (1 - y_true) * np.log(1 - y_pred + eps))


# ---------------------------------------------------------------------------
# 4. Weight initialisation
# ---------------------------------------------------------------------------
# He initialisation: scale by sqrt(2 / fan_in).
# Keeps the variance of activations stable through ReLU layers; plain
# random or zeros leads to vanishing/exploding signals in deep nets.

def init_params(n_input, n_hidden, n_output, seed=0):
    rng = np.random.default_rng(seed)
    params = {
        "W1": rng.normal(0, np.sqrt(2.0 / n_input),  (n_input,  n_hidden)),
        "b1": np.zeros((1, n_hidden)),
        "W2": rng.normal(0, np.sqrt(2.0 / n_hidden), (n_hidden, n_output)),
        "b2": np.zeros((1, n_output)),
    }
    return params


# ---------------------------------------------------------------------------
# 5. Forward pass
# ---------------------------------------------------------------------------
# We store every intermediate value (Z1, A1, Z2, A2) in a `cache` dict.
# The backward pass NEEDS these to compute gradients — without them you'd
# have to recompute the whole forward pass again for each parameter.

def forward(X, params):
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    Z1 = X  @ W1 + b1   # (n, hidden) — linear combination of inputs
    A1 = relu(Z1)        # (n, hidden) — apply non-linearity

    Z2 = A1 @ W2 + b2   # (n, 1)     — linear combination of hidden units
    A2 = sigmoid(Z2)     # (n, 1)     — squash to probability [0, 1]

    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


# ---------------------------------------------------------------------------
# 6. Backward pass — the heart of backpropagation
# ---------------------------------------------------------------------------
#
# We apply the chain rule layer by layer, right to left.
#
# ┌─────────────────────────────────────────────────────────┐
# │  FORWARD:  X  →[W1,b1]→  Z1  →[relu]→  A1             │
# │                →[W2,b2]→  Z2  →[σ]→  A2  →[BCE]→  L   │
# │                                                         │
# │  BACKWARD: L  → dA2 → dZ2 → {dW2, db2, dA1}           │
# │                     → dZ1 → {dW1, db1}                 │
# └─────────────────────────────────────────────────────────┘
#
# KEY TRICK — Sigmoid + BCE collapse to one clean expression:
#
#   L  = BCE(y, sigmoid(Z2))
#   dL/dZ2 = A2 - y   (divided by n for the mean)
#
# This avoids computing dL/dA2 and dA2/dZ2 separately; they simplify.
# Derivation: dL/dA2 = -(y/A2 - (1-y)/(1-A2)), dA2/dZ2 = A2*(1-A2)
#             product → A2 - y  ✓

def backward(y, params, cache):
    n  = y.shape[0]
    X  = cache["X"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    # --- Layer 2 (output) ---
    dZ2 = (A2 - y) / n                      # (n, 1)   — see derivation above
    dW2 = A1.T @ dZ2                         # (hidden, 1)
    db2 = dZ2.sum(axis=0, keepdims=True)     # (1, 1)

    # --- Pass error back through W2 ---
    dA1 = dZ2 @ params["W2"].T              # (n, hidden)

    # --- Layer 1 (hidden) — chain rule through ReLU ---
    # ReLU is not differentiable at 0, but we treat it as 0 there (standard).
    dZ1 = dA1 * relu_grad(Z1)               # (n, hidden) — element-wise mask
    dW1 = X.T  @ dZ1                        # (input, hidden)
    db1 = dZ1.sum(axis=0, keepdims=True)    # (1, hidden)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


# ---------------------------------------------------------------------------
# 7. Training loop
# ---------------------------------------------------------------------------

def train(X, y, n_hidden=8, lr=0.5, epochs=2000, verbose=True):
    n_input, n_output = X.shape[1], y.shape[1]
    params = init_params(n_input, n_hidden, n_output)
    history = []

    for epoch in range(epochs):
        y_pred, cache = forward(X, params)

        loss = bce_loss(y, y_pred)
        history.append(loss)

        grads = backward(y, params, cache)

        for key in params:
            params[key] -= lr * grads[key]

        if verbose and (epoch % 200 == 0 or epoch == epochs - 1):
            acc = ((y_pred >= 0.5) == y).mean()
            print(f"epoch {epoch:4d} | loss = {loss:.4f} | acc = {acc:.3f}")

    return params, history


# ---------------------------------------------------------------------------
# 8. Put it all together
# ---------------------------------------------------------------------------

def main():
    X, y = make_xor_data(n=400)

    print("Training MLP on XOR problem")
    print("Architecture: 2 → 8 (ReLU) → 1 (Sigmoid)")
    print("-" * 50)
    params, history = train(X, y, n_hidden=8, lr=0.5, epochs=2000)

    # ---- Plot: decision boundary (left) and loss curve (right) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Decision boundary — evaluate the model on a dense grid
    xs = np.linspace(-1.5, 1.5, 300)
    ys = np.linspace(-1.5, 1.5, 300)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs, _ = forward(grid, params)
    probs = probs.reshape(xx.shape)

    ax1.contourf(xx, yy, probs, levels=50, cmap="RdBu", alpha=0.75)
    ax1.contour(xx, yy, probs, levels=[0.5], colors="k", linewidths=1)
    ax1.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="RdBu",
                edgecolors="k", linewidths=0.4, s=20)
    ax1.set_title("Decision Boundary (XOR)")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")

    ax2.plot(history, "b-")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("BCE loss")
    ax2.set_title("Loss during training")
    ax2.set_yscale("log")

    fig.tight_layout()
    # Save next to this script, regardless of the current working directory.
    out = os.path.join(os.path.dirname(__file__), "mlp_backprop.png")
    fig.savefig(out, dpi=120)
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
