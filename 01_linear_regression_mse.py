"""
Example 01 — Linear Regression with Mean Squared Error (MSE)
============================================================

Goal of this lesson
-------------------
Learn the *full* mechanics of the simplest supervised-learning model by
building it from scratch with NumPy (no scikit-learn, no PyTorch). We cover:

  1. The model            :  y_hat = w * x + b               (a straight line)
  2. The loss function    :  MSE = (1/n) * sum((y - y_hat)^2)
  3. The gradients        :  dL/dw and dL/db (calculus, derived below)
  4. Gradient descent     :  iteratively nudge w and b to shrink the loss
  5. The closed-form fit  :  the exact least-squares answer (for comparison)

Run it:
    python 01_linear_regression_mse.py

It prints the learned parameters and saves a plot to `linear_regression_fit.png`.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless backend: render to a file, no GUI window needed
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Create a synthetic dataset
# ---------------------------------------------------------------------------
# We *invent* data that follows a known line  y = 2.5 * x + 7  and then add
# random noise. Because we know the "true" parameters, we can check whether
# our model recovers them. This is a classic way to sanity-check a learner.

def make_data(n_samples=100, true_w=2.5, true_b=7.0, noise=2.0, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, size=n_samples)          # feature in range [0, 10]
    noise_term = rng.normal(0, noise, size=n_samples)
    y = true_w * x + true_b + noise_term            # target with measurement noise
    return x, y


# ---------------------------------------------------------------------------
# 2. The model and the loss
# ---------------------------------------------------------------------------

def predict(x, w, b):
    """The model: a straight line. Returns y_hat for every x."""
    return w * x + b


def mse_loss(y_true, y_pred):
    """
    Mean Squared Error.

        MSE = (1/n) * sum_i (y_i - y_hat_i)^2

    We square the errors so that (a) negative and positive errors don't cancel,
    and (b) large errors are penalised much more than small ones. Lower = better.
    """
    return np.mean((y_true - y_pred) ** 2)


# ---------------------------------------------------------------------------
# 3. The gradients (the heart of "learning")
# ---------------------------------------------------------------------------
# We want to change w and b to reduce the MSE. Calculus tells us which
# direction to move. Starting from
#
#     L(w, b) = (1/n) * sum (y - (w*x + b))^2
#
# the partial derivatives are:
#
#     dL/dw = (-2/n) * sum( x * (y - y_hat) )
#     dL/db = (-2/n) * sum(     (y - y_hat) )
#
# The gradient points in the direction of *steepest increase*, so to MINIMISE
# the loss we step in the OPPOSITE direction (hence the minus sign in the update).

def gradients(x, y, w, b):
    n = len(x)
    y_hat = predict(x, w, b)
    error = y - y_hat                       # how far off each prediction is
    dw = (-2.0 / n) * np.sum(x * error)
    db = (-2.0 / n) * np.sum(error)
    return dw, db


# ---------------------------------------------------------------------------
# 4. Training loop — gradient descent
# ---------------------------------------------------------------------------

def train(x, y, lr=0.01, epochs=1000, verbose=True):
    """
    lr     : learning rate — how big a step we take each update. Too big -> diverge,
             too small -> slow. 0.01 is a sensible start for this scaled problem.
    epochs : how many full passes (updates) over the data we perform.
    """
    w, b = 0.0, 0.0                         # start from a blank slate
    history = []                            # track loss so we can plot the curve

    for epoch in range(epochs):
        dw, db = gradients(x, y, w, b)
        w -= lr * dw                        # step downhill in w
        b -= lr * db                        # step downhill in b

        loss = mse_loss(y, predict(x, w, b))
        history.append(loss)

        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:4d} | loss = {loss:8.4f} | w = {w:6.3f} | b = {b:6.3f}")

    return w, b, history


# ---------------------------------------------------------------------------
# 5. Closed-form solution (the exact least-squares answer)
# ---------------------------------------------------------------------------
# For simple linear regression there is an analytic optimum — no iteration needed.
# It's great for checking that gradient descent converged to the right place.
#
#     w = cov(x, y) / var(x)
#     b = mean(y) - w * mean(x)

def closed_form(x, y):
    x_mean, y_mean = x.mean(), y.mean()
    w = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    b = y_mean - w * x_mean
    return w, b


# ---------------------------------------------------------------------------
# 6. Put it all together
# ---------------------------------------------------------------------------

def main():
    true_w, true_b = 2.5, 7.0
    x, y = make_data(true_w=true_w, true_b=true_b)

    print("Training with gradient descent")
    print("-" * 50)
    w_gd, b_gd, history = train(x, y, lr=0.01, epochs=1000)

    w_cf, b_cf = closed_form(x, y)

    print("\nResults")
    print("-" * 50)
    print(f"True parameters          : w = {true_w:.3f}, b = {true_b:.3f}")
    print(f"Gradient descent         : w = {w_gd:.3f}, b = {b_gd:.3f}")
    print(f"Closed-form (exact) fit  : w = {w_cf:.3f}, b = {b_cf:.3f}")
    print(f"Final MSE (gradient desc): {mse_loss(y, predict(x, w_gd, b_gd)):.4f}")

    # ---- Plot: data + fitted line (left) and loss curve (right) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(x, y, alpha=0.6, label="data (noisy)")
    xs = np.linspace(x.min(), x.max(), 100)
    ax1.plot(xs, predict(xs, w_gd, b_gd), "r-", linewidth=2,
             label=f"fit: y = {w_gd:.2f}x + {b_gd:.2f}")
    ax1.plot(xs, predict(xs, true_w, true_b), "g--", linewidth=2,
             label=f"truth: y = {true_w}x + {true_b}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Linear Regression Fit")
    ax1.legend()

    ax2.plot(history, "b-")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("MSE loss")
    ax2.set_title("Loss decreasing during training")
    ax2.set_yscale("log")  # log scale makes the early rapid drop easier to read

    fig.tight_layout()
    out = "linear_regression_fit.png"
    fig.savefig(out, dpi=120)
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
