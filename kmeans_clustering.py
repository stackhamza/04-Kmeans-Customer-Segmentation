"""
K-Means Clustering - Customer Segmentation
Algorithm: K-Means (unsupervised learning)
Dataset: Synthetic customer data (annual income vs spending score)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def make_customer_data(n_samples=400, random_state=42):
    """Generate synthetic customer data with 4 natural segments."""
    rng = np.random.RandomState(random_state)
    centers = [
        (25, 20),   # low income, low spending
        (25, 80),   # low income, high spending (impulsive)
        (80, 20),   # high income, low spending (savers)
        (80, 80),   # high income, high spending (premium)
    ]
    per_cluster = n_samples // len(centers)
    income, spending = [], []
    for cx, cy in centers:
        income.extend(rng.normal(cx, 8, per_cluster))
        spending.extend(rng.normal(cy, 8, per_cluster))
    return np.column_stack([income, spending])


def main():
    # 1. Data
    X = make_customer_data()
    print("Dataset shape:", X.shape)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Elbow method to choose k
    inertias = []
    k_range = range(1, 10)
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    # 3. Fit final model with k=4
    k_final = 4
    model = KMeans(n_clusters=k_final, n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    print(f"\nChosen k = {k_final}")
    print(f"Silhouette Score: {sil:.4f}")

    centers_orig = scaler.inverse_transform(model.cluster_centers_)
    for i, c in enumerate(centers_orig):
        print(f"  Cluster {i}: income≈{c[0]:.1f}k, spending_score≈{c[1]:.1f} "
              f"({np.sum(labels == i)} customers)")

    # 4. Plots: elbow curve + cluster scatter
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(list(k_range), inertias, "o-", color="#2563eb")
    axes[0].axvline(k_final, color="red", linestyle="--", label=f"chosen k={k_final}")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method for Optimal k")
    axes[0].legend()

    colors = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626"]
    for i in range(k_final):
        mask = labels == i
        axes[1].scatter(X[mask, 0], X[mask, 1], s=25, alpha=0.6,
                         color=colors[i % len(colors)], label=f"Segment {i}")
    axes[1].scatter(centers_orig[:, 0], centers_orig[:, 1], marker="X", s=250,
                     color="black", label="Centroids")
    axes[1].set_xlabel("Annual Income (k$)")
    axes[1].set_ylabel("Spending Score (1-100)")
    axes[1].set_title(f"Customer Segments (Silhouette = {sil:.3f})")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("screenshots/results.png", dpi=150)
    print("\nSaved plot to screenshots/results.png")


if __name__ == "__main__":
    main()
