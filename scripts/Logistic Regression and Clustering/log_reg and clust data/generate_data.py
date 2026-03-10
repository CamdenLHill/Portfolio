"""
AERO 689: Homework 3 – Data Generator
Classification and Clustering

Run this script once to generate all datasets needed for HW3.

Usage:
    python generate_data.py

Outputs (all saved to this directory):
    p1a_sensor_binary.csv      – Part 1a: Binary classification (normal vs degraded)
    p1b_flight_phases.csv      – Part 1b: Multiclass classification (3 flight phases)
    p1c_regularization.csv     – Part 1c: Regularization and noisy data
    p2a_operating_modes.csv    – Part 2a: Well-behaved k-means (3 spherical clusters)
    p2b_choose_k.csv           – Part 2b: Elbow / silhouette analysis (choose k)
    p2c_scaling.csv            – Part 2c: K-means with feature scaling problem
    p3a_nonlinear.csv          – Part 3a: Non-linear decision boundary
    p3b_two_moons.csv          – Part 3b: Non-spherical clusters (k-means failure)
    p3c_densities.csv          – Part 3c: Clusters with different densities
    p3d_init.csv               – Part 3d: Initialization sensitivity demo
    p4_integration.csv         – Part 4: Integration problem (high-dim sensor data)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_blobs

# ── Fixed seed for reproducibility ────────────────────────────────────────────
RNG = np.random.default_rng(42)


# ─────────────────────────────────────────────────────────────────────────────
# Part 1a: Binary classification – Normal vs degraded engine sensor readings
#   Features: vibration_rms (g),  exhaust_temp_deviation (°C)
#   Label:    0 = normal, 1 = degraded
# ─────────────────────────────────────────────────────────────────────────────
def gen_p1a(n=300):
    # Normal: low vibration, small temp deviation
    n0 = n // 2
    X0 = RNG.multivariate_normal(
        mean=[1.2, 10.0],
        cov=[[0.10, 0.05], [0.05, 8.0]],
        size=n0,
    )
    # Degraded: higher vibration, larger temp deviation
    n1 = n - n0
    X1 = RNG.multivariate_normal(
        mean=[3.5, 45.0],
        cov=[[0.25, 0.10], [0.10, 12.0]],
        size=n1,
    )
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n0), np.ones(n1)]).astype(int)

    df = pd.DataFrame(
        {
            "vibration_rms": X[:, 0],
            "exhaust_temp_deviation": X[:, 1],
            "label": y,
        }
    )
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part 1b: Multiclass – Three flight phases
#   Features: altitude_rate (ft/min),  airspeed_deviation (kts)
#   Labels:   0 = descent, 1 = cruise, 2 = climb
# ─────────────────────────────────────────────────────────────────────────────
def gen_p1b(n=360):
    n_each = n // 3

    # Descent: negative altitude rate, slightly slower airspeed
    X0 = RNG.multivariate_normal(
        mean=[-1500, -15.0],
        cov=[[120000, 200], [200, 20]],
        size=n_each,
    )
    # Cruise: near-zero altitude rate, near-zero airspeed deviation
    X1 = RNG.multivariate_normal(
        mean=[0, 0.0],
        cov=[[8000, 50], [50, 8]],
        size=n_each,
    )
    # Climb: positive altitude rate, slightly higher airspeed
    X2 = RNG.multivariate_normal(
        mean=[1500, 12.0],
        cov=[[130000, 300], [300, 18]],
        size=n_each,
    )

    X = np.vstack([X0, X1, X2])
    y = np.hstack(
        [np.zeros(n_each), np.ones(n_each), 2 * np.ones(n_each)]
    ).astype(int)

    # Add a small number of overlapping samples near boundaries to keep it interesting
    noise_X = RNG.multivariate_normal([0, 0], [[50000, 0], [0, 10]], size=15)
    noise_y = RNG.integers(0, 3, size=15)
    X = np.vstack([X, noise_X])
    y = np.hstack([y, noise_y])

    df = pd.DataFrame(
        {
            "altitude_rate": X[:, 0],
            "airspeed_deviation": X[:, 1],
            "phase": y,
        }
    )
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part 1c: Regularization – binary sensor data with noise + outliers
#   Two classes that are mostly separable but with noise and a few outliers
#   Students will compare unregularized vs L2 logistic regression
# ─────────────────────────────────────────────────────────────────────────────
def gen_p1c(n=250):
    # Class 0
    n0 = n // 2
    X0 = RNG.multivariate_normal(
        mean=[2.0, 2.0], cov=[[0.8, 0.3], [0.3, 0.8]], size=n0
    )
    # Class 1
    n1 = n - n0
    X1 = RNG.multivariate_normal(
        mean=[5.0, 5.0], cov=[[0.8, 0.3], [0.3, 0.8]], size=n1
    )
    # Outliers: class-0 points placed deep in class-1 territory
    outliers_X = RNG.multivariate_normal(
        mean=[5.5, 5.5], cov=[[0.1, 0.0], [0.0, 0.1]], size=8
    )
    outliers_y = np.zeros(8, dtype=int)  # class 0, but in class-1 space

    X = np.vstack([X0, X1, outliers_X])
    y = np.hstack([np.zeros(n0), np.ones(n1), outliers_y]).astype(int)

    df = pd.DataFrame({"feature_1": X[:, 0], "feature_2": X[:, 1], "label": y})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part 2a: Well-behaved k-means – three clear operating modes
#   (e.g., three engine operating regimes: idle, cruise power, max-continuous)
# ─────────────────────────────────────────────────────────────────────────────
def gen_p2a(n=300):
    centers = np.array([[1.0, 1.0], [5.0, 5.0], [9.0, 1.5]])
    stds = [0.5, 0.5, 0.5]
    X, y = make_blobs(
        n_samples=n,
        centers=centers,
        cluster_std=stds,
        random_state=42,
    )
    df = pd.DataFrame(
        {"fuel_flow_normalized": X[:, 0], "thrust_normalized": X[:, 1], "true_mode": y}
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part 2b: Choosing k – 4 natural clusters (students must discover k=4)
# ─────────────────────────────────────────────────────────────────────────────
def gen_p2b(n=400):
    centers = np.array(
        [[0.0, 0.0], [7.0, 0.0], [0.0, 7.0], [7.0, 7.0]]
    )
    X, y = make_blobs(
        n_samples=n,
        centers=centers,
        cluster_std=1.0,
        random_state=42,
    )
    df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1]})# , "label": y})
    # No true_label column – students must discover using elbow + silhouette
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part 2c: Feature scaling problem
#   Two operational states (normal=0, degraded=1) distinguished by temperature.
#   Oil pressure varies randomly – it carries NO cluster information – but its
#   raw scale (hundreds of psi) utterly dwarfs temperature (tens of °C).
#   Without scaling: k-means cuts by oil pressure (random → ARI≈0).
#   With scaling:    temperature cleanly separates both states (→ ARI≈1).
# ─────────────────────────────────────────────────────────────────────────────
def gen_p2c(n=300):
    n_each = n // 2
    # Normal (0): cool running,  temp ~25°C
    # Degraded (1): hot running, temp ~75°C
    # Oil pressure is the same distribution for BOTH states (pure noise)
    X0 = RNG.multivariate_normal(
        mean=[25.0, 1400.0], cov=[[16, 0], [0, 160000]], size=n_each
    )
    X1 = RNG.multivariate_normal(
        mean=[75.0, 1400.0], cov=[[16, 0], [0, 160000]], size=n_each
    )
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_each), np.ones(n_each)]).astype(int)
    df = pd.DataFrame(
        {"temperature_C": X[:, 0], "oil_pressure_psi": X[:, 1], "true_state": y}
    )
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part 3a: Non-linear decision boundary – concentric rings
#   Two classes arranged in concentric circles (inner vs outer ring)
# ─────────────────────────────────────────────────────────────────────────────
def gen_p3a(n=400):
    from sklearn.datasets import make_circles

    X, y = make_circles(n_samples=n, noise=0.10, factor=0.45, random_state=42)
    df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "label": y})
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part 3b: Two moons – k-means failure on non-spherical clusters
# ─────────────────────────────────────────────────────────────────────────────
def gen_p3b(n=400):
    X, y = make_moons(n_samples=n, noise=0.08, random_state=42)
    df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "true_cluster": y})
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part 3c: Different cluster densities
#   Three clusters with very different standard deviations:
#     Cluster 0: tight (σ=0.4)
#     Cluster 1: medium (σ=1.2)
#     Cluster 2: spread (σ=2.8)
# ─────────────────────────────────────────────────────────────────────────────
def gen_p3c(n=450):
    n_each = n // 3
    X0 = RNG.multivariate_normal([0.0, 0.0], np.eye(2) * 0.4**2, size=n_each)
    X1 = RNG.multivariate_normal([8.0, 0.0], np.eye(2) * 1.2**2, size=n_each)
    X2 = RNG.multivariate_normal([4.0, 8.0], np.eye(2) * 2.8**2, size=n_each)
    X = np.vstack([X0, X1, X2])
    y = np.hstack(
        [np.zeros(n_each), np.ones(n_each), 2 * np.ones(n_each)]
    ).astype(int)
    df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "true_cluster": y})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part 3d: Initialization sensitivity – well-separated but far enough apart
#   that bad initializations miss a cluster
# ─────────────────────────────────────────────────────────────────────────────
def gen_p3d(n=300):
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 9.0]])
    X, y = make_blobs(
        n_samples=n, centers=centers, cluster_std=1.0, random_state=42
    )
    df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "true_cluster": y})
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Part 4: Integration problem – 6-dimensional sensor data
#   Simulates aircraft engine health monitoring with 6 sensor channels.
#   True structure: 3 underlying health states (healthy, early-fault, severe-fault)
#   Clusters are well-separated (~3-4 sigma on primary features) so k-means can
#   recover them reliably (ARI > 0.85) after standardization.
#   But students do NOT see labels initially.
# ─────────────────────────────────────────────────────────────────────────────
def gen_p4(n=360):
    n_each = n // 3

    # Healthy: all sensors at baseline
    X0 = RNG.multivariate_normal(
        mean=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        cov=np.diag([0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
        size=n_each,
    )
    # Early fault: elevated vibration (features 0,1), moderate temp rise (feature 4)
    X1 = RNG.multivariate_normal(
        mean=[2.5, 2.5, 0.5, 0.5, 1.5, 0.3],
        cov=np.diag([0.35, 0.35, 0.35, 0.35, 0.35, 0.35]),
        size=n_each,
    )
    # Severe fault: very high vibration, high temp, large oil pressure deviation
    X2 = RNG.multivariate_normal(
        mean=[5.5, 5.5, 1.2, 1.5, 4.0, 1.2],
        cov=np.diag([0.4, 0.4, 0.4, 0.4, 0.4, 0.4]),
        size=n_each,
    )

    X = np.vstack([X0, X1, X2])
    y = np.hstack(
        [np.zeros(n_each), np.ones(n_each), 2 * np.ones(n_each)]
    ).astype(int)

    col_names = [
        "vib_channel_1",
        "vib_channel_2",
        "accel_x",
        "accel_y",
        "exhaust_temp",
        "oil_pressure_dev",
    ]
    df_no_label = pd.DataFrame(X, columns=col_names)
    df_with_label = df_no_label.copy()
    df_with_label["health_state"] = y

    df_no_label = df_no_label.sample(frac=1, random_state=42).reset_index(drop=True)
    df_with_label = df_with_label.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_no_label, df_with_label


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    out_dir = os.path.dirname(os.path.abspath(__file__))

    datasets = {
        "p1a_sensor_binary.csv":    gen_p1a(),
        "p1b_flight_phases.csv":    gen_p1b(),
        "p1c_regularization.csv":   gen_p1c(),
        "p2a_operating_modes.csv":  gen_p2a(),
        "p2b_choose_k.csv":         gen_p2b(),
        "p2c_scaling.csv":          gen_p2c(),
        "p3a_nonlinear.csv":        gen_p3a(),
        "p3b_two_moons.csv":        gen_p3b(),
        "p3c_densities.csv":        gen_p3c(),
        "p3d_init.csv":             gen_p3d(),
    }

    for fname, df in datasets.items():
        path = os.path.join(out_dir, fname)
        df.to_csv(path, index=False)
        print(f"  Saved {fname:35s}  shape={df.shape}")

    # Part 4 has two variants
    df_p4, df_p4_labels = gen_p4()
    df_p4.to_csv(os.path.join(out_dir, "p4_integration.csv"), index=False)
    df_p4_labels.to_csv(os.path.join(out_dir, "p4_integration_labels.csv"), index=False)
    print(f"  Saved p4_integration.csv             shape={df_p4.shape}")
    print(f"  Saved p4_integration_labels.csv      shape={df_p4_labels.shape}  (used in Problem 6 for ARI evaluation)")

    print("\nAll datasets generated successfully.")
