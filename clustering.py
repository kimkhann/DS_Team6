"""
================================================================
Extract attack traffic -> PCA -> K-Means clustering -> Evaluation -> Visualization

Input files (from Seunghoon's preprocessing):
    X_train.csv          : Feature data (RobustScaler applied, 182 dims)
    y_train.csv          : Binary label (0 = normal, 1 = attack)
    y_train_cat.csv      : Attack category (used as ground truth for evaluation)

Output files:
    01_elbow_curve.png       : Elbow plot to find optimal k
    02_silhouette_score.png  : Silhouette score plot
    03_pca_2d_scatter.png    : 2D PCA cluster visualization
    04_crosstab_heatmap.png  : Cluster vs. attack category comparison
    clustering_report.txt    : Log of all results
================================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Save figures without opening a window
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
import warnings
warnings.filterwarnings('ignore')

# Prevent minus sign from breaking in plots
plt.rcParams['axes.unicode_minus'] = False

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(str(msg))


# Configuration (all settings in one place)
RANDOM_STATE = 42       # Seed for reproducibility
K_RANGE = range(2, 16)  # Search k from 2 to 15
FINAL_K = 8             # Final number of clusters (8 real attack categories)
PCA_DIM = 20            # Fixed PCA output dim (prevents dimension collapse from outliers)


# Step 1. Load data and extract attack records only
def load_attack_data():
    """
    Keep only attack traffic (label = 1).
    Clustering aims to analyze attack patterns, so normal traffic (label = 0)
    is not used here.
    """
    log("=" * 60)
    log("[Step 1] Load data and extract attack records")
    log("=" * 60)

    X = pd.read_csv('X_train.csv')
    y = pd.read_csv('y_train.csv').squeeze().reset_index(drop=True)
    y_cat = pd.read_csv('y_train_cat.csv').squeeze().reset_index(drop=True)

    log(f"  Full data: {X.shape[0]:,} rows x {X.shape[1]} cols")

    # Strip leading spaces in category names (e.g., ' Fuzzers' -> 'Fuzzers')
    y_cat = y_cat.astype(str).str.strip()

    # Select rows where label == 1 (attack) using boolean mask
    attack_mask = (y == 1).values
    X_attack = X[attack_mask].reset_index(drop=True)
    y_cat_attack = y_cat[attack_mask].reset_index(drop=True)

    log(f"  Attack data: {X_attack.shape[0]:,} rows x {X_attack.shape[1]} cols")
    log(f"\n  Attack category distribution:")
    for cat, cnt in y_cat_attack.value_counts().items():
        log(f"    {cat:20s} {cnt:>6,}")
    log(f"\n  Number of categories: {y_cat_attack.nunique()}")

    return X_attack, y_cat_attack


# Step 2. PCA dimensionality reduction
def apply_pca(X):
    """
    Reduce dimensions from 182 to 20 using PCA.

    [Extra step: log1p transform]
      Even after RobustScaler, some features (sbytes, sload, etc.) still hold
      extreme values. Without log1p, PCA variance concentrates on those few
      extreme directions and the first 2-3 components absorb almost all variance,
      causing dimension collapse. log1p (= log(1 + x)) compresses large values
      while keeping small ones, which makes the variance more balanced.

    [Fixed dim: PCA_DIM = 20]
      Using n_components = 0.95 auto-selects only 3 dims because of the outlier
      issue above. 3 dims is too few to separate 8 attack categories, so we
      fix the output to 20 dims, which is enough to preserve useful structure.
    """
    log("\n" + "=" * 60)
    log("[Step 2] PCA dimensionality reduction")
    log("=" * 60)

    # Sign-preserving log transform for negative values: sign(x) * log1p(|x|)
    X_log = np.sign(X) * np.log1p(np.abs(X))

    pca = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_log)

    log(f"  Original dim: {X.shape[1]}")
    log(f"  log1p transform applied (reduces outlier effect)")
    log(f"  After PCA: {X_pca.shape[1]} dims (fixed)")
    log(f"  Cumulative explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    return X_pca, pca


# Step 3. Find optimal k (Elbow Method + Silhouette Score)
def find_optimal_k(X_pca):
    """
    K-Means needs k to be set in advance. We try two methods.

    [Method 1] Elbow Method
      - Measure inertia (sum of squared distances inside clusters) for each k
      - As k grows, inertia drops; the point where the drop slows down is best k

    [Method 2] Silhouette Score
      - For each point, score how close it is to its own cluster vs. others
      - Range: -1 to 1. Higher is better. Best k = highest score
    """
    log("\n" + "=" * 60)
    log("[Step 3] Find optimal k (k=2 to 15)")
    log("=" * 60)

    inertias = []        # For Elbow
    silhouettes = []     # For Silhouette

    # Silhouette is expensive, so compute it on a 5000-row sample only
    sample_size = min(5000, len(X_pca))
    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(len(X_pca), sample_size, replace=False)
    X_sample = X_pca[sample_idx]

    log(f"  Silhouette sample size: {sample_size:,}")
    log(f"\n  {'k':>4s} | {'inertia':>15s} | {'silhouette':>12s}")
    log(f"  {'-'*4} | {'-'*15} | {'-'*12}")

    for k in K_RANGE:
        # K-Means with n_init=10: try 10 random starts and keep the best one
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels_full = km.fit_predict(X_pca)

        # inertia: sum of squared distances from each point to its cluster center
        inertias.append(km.inertia_)

        # silhouette: computed on the sample for speed
        labels_sample = labels_full[sample_idx]
        sil = silhouette_score(X_sample, labels_sample)
        silhouettes.append(sil)

        log(f"  {k:>4d} | {km.inertia_:>15,.0f} | {sil:>12.4f}")

    # k with the highest silhouette score
    best_sil_k = list(K_RANGE)[np.argmax(silhouettes)]
    log(f"\n  -> Best silhouette k: {best_sil_k} (score: {max(silhouettes):.4f})")
    log(f"  -> Domain-based k: {FINAL_K} (number of real attack categories)")

    return inertias, silhouettes, best_sil_k


# Step 4. Final K-Means clustering
def final_clustering(X_pca, k):
    """
    Run K-Means with the chosen k and return cluster labels for every point.
    """
    log("\n" + "=" * 60)
    log(f"[Step 4] Final K-Means clustering (k={k})")
    log("=" * 60)

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_pca)

    log(f"  Cluster size distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        log(f"    Cluster {u}: {c:,}")

    return labels, km


# Step 5. Evaluation (compare against real attack categories)
def evaluate(labels, y_cat_attack):
    """
    Clustering is unsupervised, but since we have the real labels (attack_cat),
    we can measure how close the clusters are to the true categories.

    [ARI - Adjusted Rand Index]
      - Range: -1 to 1. Near 0 = random, 1 = perfect match
      - Corrects for chance agreement

    [NMI - Normalized Mutual Information]
      - Range: 0 to 1. 0 = no shared info, 1 = perfect match
      - Based on information theory
    """
    log("\n" + "=" * 60)
    log("[Step 5] Evaluation - compare with real attack categories")
    log("=" * 60)

    ari = adjusted_rand_score(y_cat_attack, labels)
    nmi = normalized_mutual_info_score(y_cat_attack, labels)

    log(f"  ARI (Adjusted Rand Index):           {ari:.4f}")
    log(f"  NMI (Normalized Mutual Information): {nmi:.4f}")

    # Cross-tabulation: cluster vs. real category
    crosstab = pd.crosstab(
        pd.Series(labels, name='Cluster'),
        pd.Series(y_cat_attack.values, name='AttackCategory')
    )
    log(f"\n  Cross-tabulation (real attack type distribution per cluster):")
    log(crosstab.to_string())

    return ari, nmi, crosstab


# Plot 1: Elbow Curve
def plot_elbow(inertias):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(K_RANGE), inertias, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax.axvline(x=FINAL_K, color='red', linestyle='--', alpha=0.6, label=f'Chosen k={FINAL_K}')
    ax.set_xlabel('Number of clusters (k)', fontsize=12)
    ax.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
    ax.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    ax.set_xticks(list(K_RANGE))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('01_elbow_curve.png', dpi=120, bbox_inches='tight')
    plt.close()
    log("  -> 01_elbow_curve.png saved")


# Plot 2: Silhouette Score
def plot_silhouette(silhouettes):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(K_RANGE), silhouettes, marker='s', linewidth=2, markersize=8, color='#A23B72')
    ax.axvline(x=FINAL_K, color='red', linestyle='--', alpha=0.6, label=f'Chosen k={FINAL_K}')
    ax.set_xlabel('Number of clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Score by k (higher is better)', fontsize=14, fontweight='bold')
    ax.set_xticks(list(K_RANGE))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('02_silhouette_score.png', dpi=120, bbox_inches='tight')
    plt.close()
    log("  -> 02_silhouette_score.png saved")


# Plot 3: PCA 2D Scatter
def plot_pca_2d(X_attack, labels, y_cat_attack):
    """
    Visualize clusters in 2D. This uses a separate 2D PCA just for plotting
    (different from the 20D PCA used for clustering).
    """
    # Plotting-only 2D PCA (log1p applied to reduce outlier effect)
    X_log = np.sign(X_attack) * np.log1p(np.abs(X_attack))
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca_2d.fit_transform(X_log)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: colored by K-Means cluster label
    scatter1 = axes[0].scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=labels, cmap='tab10', s=8, alpha=0.6
    )
    axes[0].set_title(f'K-Means Clusters (k={FINAL_K})', fontsize=13, fontweight='bold')
    axes[0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')

    # Right: colored by real attack category
    categories = sorted(y_cat_attack.unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    for cat, color in zip(categories, colors):
        mask = (y_cat_attack == cat).values
        axes[1].scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[color], label=cat, s=8, alpha=0.6
        )
    axes[1].set_title('Actual Attack Categories', fontsize=13, fontweight='bold')
    axes[1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1].legend(loc='best', fontsize=9, markerscale=2)

    plt.tight_layout()
    plt.savefig('03_pca_2d_scatter.png', dpi=120, bbox_inches='tight')
    plt.close()
    log("  -> 03_pca_2d_scatter.png saved")


# Plot 4: Cross-tabulation Heatmap
def plot_crosstab_heatmap(crosstab):
    """
    Heatmap of cluster (rows) vs. real attack category (cols).
    The more a cluster is dominated by one category, the better the clustering.
    """
    # Normalize: ratio of each category within each cluster (row)
    crosstab_norm = crosstab.div(crosstab.sum(axis=1), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Left: raw counts
    sns.heatmap(
        crosstab, annot=True, fmt='d', cmap='Blues',
        cbar_kws={'label': 'Count'}, ax=axes[0]
    )
    axes[0].set_title('Cluster vs Attack Category (Counts)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Attack Category')
    axes[0].set_ylabel('Cluster')

    # Right: row-normalized ratios
    sns.heatmap(
        crosstab_norm, annot=True, fmt='.2f', cmap='YlOrRd',
        cbar_kws={'label': 'Proportion'}, ax=axes[1], vmin=0, vmax=1
    )
    axes[1].set_title('Cluster Composition (Row-normalized)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Attack Category')
    axes[1].set_ylabel('Cluster')

    plt.tight_layout()
    plt.savefig('04_crosstab_heatmap.png', dpi=120, bbox_inches='tight')
    plt.close()
    log("  -> 04_crosstab_heatmap.png saved")


# Main
if __name__ == "__main__":
    # 1. Load data
    X_attack, y_cat_attack = load_attack_data()

    # 2. PCA reduction
    X_pca, pca_model = apply_pca(X_attack)

    # 3. Find optimal k
    inertias, silhouettes, best_sil_k = find_optimal_k(X_pca)

    # 4. Final clustering (k=8, matches the number of real attack categories)
    labels, km_model = final_clustering(X_pca, FINAL_K)

    # 5. Evaluation
    ari, nmi, crosstab = evaluate(labels, y_cat_attack)

    # 6. Visualization
    log("\n" + "=" * 60)
    log("[Step 6] Save plots")
    log("=" * 60)
    plot_elbow(inertias)
    plot_silhouette(silhouettes)
    plot_pca_2d(X_attack.values, labels, y_cat_attack)
    plot_crosstab_heatmap(crosstab)

    # 7. Save log file
    with open('clustering_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

    log("\n" + "=" * 60)
    log("Clustering done")
    log("=" * 60)
    log(f"  Final k: {FINAL_K}")
    log(f"  ARI: {ari:.4f} | NMI: {nmi:.4f}")
    log(f"  Output files: 01~04*.png, clustering_report.txt")
