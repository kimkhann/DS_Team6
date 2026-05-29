"""
================================================================
Extract attack traffic -> PCA -> DBSCAN clustering -> Evaluation -> Visualization

** Comparison counterpart to clustering.py (K-Means version) **
Same input data and same preprocessing (log1p + PCA 20 dims) are used,
so that K-Means and DBSCAN can be compared under identical conditions.
The only difference is the clustering algorithm itself.

Input files (from Seunghoon's preprocessing):
    X_train.csv          : Feature data (RobustScaler applied, 182 dims)
    y_train.csv          : Binary label (0 = normal, 1 = attack)
    y_train_cat.csv      : Attack category (used as ground truth for evaluation)

Output files:
    dbscan_01_k_distance.png    : k-distance graph to help choose eps
    dbscan_02_pca_2d_scatter.png: 2D PCA cluster visualization
    dbscan_03_crosstab_heatmap.png: Cluster vs. attack category comparison
    dbscan_report.txt           : Log of all results
================================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Save figures without opening a window
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
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

X_pca_global = None  # set in main, used by evaluate() for silhouette


# Configuration (kept identical to the K-Means version where shared)
RANDOM_STATE = 42       # Seed for reproducibility
PCA_DIM = 20            # Fixed PCA output dim (same as K-Means version)

# DBSCAN-specific parameters
# min_samples: a common rule of thumb is (2 * number_of_dimensions),
#   so for 20 PCA dims we use 40. Higher values make clusters denser/cleaner.
# eps: chosen from the k-distance graph (see Step 3). EPS_CANDIDATES are swept
#   so we can report how the result changes with eps.
MIN_SAMPLES = 40
EPS_CANDIDATES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
FINAL_EPS = 2.5         # Updated automatically after the sweep (see main)


# Step 1. Load data and extract attack records only
#   (identical to the K-Means version so both run on the same data)
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
#   (identical to the K-Means version: log1p + PCA to 20 dims)
def apply_pca(X):
    """
    Reduce dimensions from 182 to 20 using PCA, after a sign-preserving log1p
    transform. This is exactly the same preprocessing used in the K-Means
    version, so the two algorithms are compared on the same feature space.
    """
    log("\n" + "=" * 60)
    log("[Step 2] PCA dimensionality reduction")
    log("=" * 60)

    X_log = np.sign(X) * np.log1p(np.abs(X))

    pca = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_log)

    log(f"  Original dim: {X.shape[1]}")
    log(f"  log1p transform applied (reduces outlier effect)")
    log(f"  After PCA: {X_pca.shape[1]} dims (fixed)")
    log(f"  Cumulative explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    return X_pca, pca


# Step 3. Help choose eps via k-distance graph + sweep eps candidates
def find_optimal_eps(X_pca):
    """
    DBSCAN does not take a cluster count. Instead it needs:
      - eps: neighborhood radius
      - min_samples: minimum neighbors to be a 'core' point

    [Method 1] k-distance graph
      For each point, measure the distance to its (min_samples)-th nearest
      neighbor, sort these distances, and plot them. The 'knee' of the curve
      is a common heuristic for a good eps.

    [Method 2] eps sweep
      Try several eps values and report, for each, the number of clusters
      found and the proportion of points labeled as noise. This shows how
      sensitive DBSCAN is to eps in this high-dimensional space.
    """
    log("\n" + "=" * 60)
    log("[Step 3] Find eps (k-distance graph + eps sweep)")
    log("=" * 60)

    # --- k-distance graph ---
    nn = NearestNeighbors(n_neighbors=MIN_SAMPLES)
    nn.fit(X_pca)
    distances, _ = nn.kneighbors(X_pca)
    # distance to the MIN_SAMPLES-th neighbor (last column), sorted ascending
    k_dist = np.sort(distances[:, -1])

    log(f"  min_samples = {MIN_SAMPLES}")
    log(f"  k-distance range: {k_dist.min():.3f} ~ {k_dist.max():.3f}")

    # --- eps sweep ---
    log(f"\n  {'eps':>6s} | {'clusters':>9s} | {'noise':>8s} | {'noise %':>8s}")
    log(f"  {'-'*6} | {'-'*9} | {'-'*8} | {'-'*8}")

    sweep_results = []
    for eps in EPS_CANDIDATES:
        db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES, n_jobs=-1)
        labels = db.fit_predict(X_pca)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        noise_pct = n_noise / len(labels) * 100
        sweep_results.append((eps, n_clusters, n_noise, noise_pct))
        log(f"  {eps:>6.1f} | {n_clusters:>9d} | {n_noise:>8,} | {noise_pct:>7.1f}%")

    return k_dist, sweep_results


# Step 4. Final DBSCAN clustering
def final_clustering(X_pca, eps):
    """
    Run DBSCAN with the chosen eps and min_samples.
    Note: DBSCAN labels noise points as -1 (not a real cluster).
    """
    log("\n" + "=" * 60)
    log(f"[Step 4] Final DBSCAN clustering (eps={eps}, min_samples={MIN_SAMPLES})")
    log("=" * 60)

    db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES, n_jobs=-1)
    labels = db.fit_predict(X_pca)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())

    log(f"  Number of clusters found: {n_clusters}")
    log(f"  Noise points (label -1): {n_noise:,} ({n_noise/len(labels)*100:.1f}%)")
    log(f"\n  Cluster size distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        name = "Noise" if u == -1 else f"Cluster {u}"
        log(f"    {name}: {c:,}")

    return labels, db


# Step 5. Evaluation (same metrics as the K-Means version)
def evaluate(labels, y_cat_attack):
    """
    Same external metrics as the K-Means version (ARI, NMI, cross-tab), so
    the two algorithms can be compared directly.

    Note on noise: DBSCAN assigns -1 to noise points. We report metrics two ways:
      (1) all points (noise treated as its own group)
      (2) excluding noise (only points that were assigned to a real cluster)
    Silhouette is computed only on non-noise points and only if at least
    2 clusters exist.
    """
    log("\n" + "=" * 60)
    log("[Step 5] Evaluation - compare with real attack categories")
    log("=" * 60)

    y_true = y_cat_attack.values

    # (1) all points
    ari_all = adjusted_rand_score(y_true, labels)
    nmi_all = normalized_mutual_info_score(y_true, labels)
    log(f"  [All points, noise as its own group]")
    log(f"    ARI: {ari_all:.4f}")
    log(f"    NMI: {nmi_all:.4f}")

    # (2) excluding noise
    mask = labels != -1
    n_real_clusters = len(set(labels[mask])) if mask.sum() > 0 else 0
    if mask.sum() > 0 and n_real_clusters >= 1:
        ari_clean = adjusted_rand_score(y_true[mask], labels[mask])
        nmi_clean = normalized_mutual_info_score(y_true[mask], labels[mask])
        log(f"  [Excluding noise: {mask.sum():,} points kept]")
        log(f"    ARI: {ari_clean:.4f}")
        log(f"    NMI: {nmi_clean:.4f}")
    else:
        ari_clean, nmi_clean = float('nan'), float('nan')
        log(f"  [Excluding noise] not enough non-noise points to evaluate")

    # Silhouette (non-noise only, needs >= 2 clusters)
    if n_real_clusters >= 2:
        # sample for speed, consistent with K-Means version (5000)
        idx = np.where(mask)[0]
        sample_size = min(5000, len(idx))
        rng = np.random.RandomState(RANDOM_STATE)
        sample = rng.choice(idx, sample_size, replace=False)
        try:
            sil = silhouette_score(X_pca_global[sample], labels[sample])
            log(f"  Silhouette (non-noise, sampled): {sil:.4f}")
        except Exception as e:
            log(f"  Silhouette could not be computed: {e}")

    # Cross-tabulation: cluster (incl. noise) vs. real category
    crosstab = pd.crosstab(
        pd.Series(labels, name='Cluster'),
        pd.Series(y_true, name='AttackCategory')
    )
    log(f"\n  Cross-tabulation (real attack type distribution per cluster):")
    log(crosstab.to_string())

    return ari_all, nmi_all, ari_clean, nmi_clean, crosstab


# Plot 1: k-distance graph
def plot_k_distance(k_dist):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(k_dist)), k_dist, linewidth=2, color='#2E86AB')
    ax.axhline(y=FINAL_EPS, color='red', linestyle='--', alpha=0.6,
               label=f'Chosen eps={FINAL_EPS}')
    ax.set_xlabel('Points sorted by distance', fontsize=12)
    ax.set_ylabel(f'Distance to {MIN_SAMPLES}-th nearest neighbor', fontsize=12)
    ax.set_title('k-distance Graph for choosing eps', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('dbscan_01_k_distance.png', dpi=120, bbox_inches='tight')
    plt.close()
    log("  -> dbscan_01_k_distance.png saved")


# Plot 2: PCA 2D Scatter (same style as K-Means version)
def plot_pca_2d(X_attack, labels, y_cat_attack):
    """
    Visualize clusters in 2D. Noise points (label -1) are drawn in light gray.
    """
    X_log = np.sign(X_attack) * np.log1p(np.abs(X_attack))
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca_2d.fit_transform(X_log)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: DBSCAN cluster labels (noise in gray)
    noise_mask = labels == -1
    axes[0].scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1],
                    c='lightgray', s=6, alpha=0.4, label='Noise')
    clustered = ~noise_mask
    if clustered.sum() > 0:
        sc = axes[0].scatter(X_2d[clustered, 0], X_2d[clustered, 1],
                             c=labels[clustered], cmap='tab10', s=8, alpha=0.6)
        plt.colorbar(sc, ax=axes[0], label='Cluster')
    axes[0].set_title(f'DBSCAN Clusters (eps={FINAL_EPS}, min_samples={MIN_SAMPLES})',
                      fontsize=13, fontweight='bold')
    axes[0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].legend(loc='best', fontsize=9, markerscale=2)

    # Right: real attack categories (same as K-Means version)
    categories = sorted(y_cat_attack.unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    for cat, color in zip(categories, colors):
        m = (y_cat_attack == cat).values
        axes[1].scatter(X_2d[m, 0], X_2d[m, 1], c=[color], label=cat, s=8, alpha=0.6)
    axes[1].set_title('Actual Attack Categories', fontsize=13, fontweight='bold')
    axes[1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1].legend(loc='best', fontsize=9, markerscale=2)

    plt.tight_layout()
    plt.savefig('dbscan_02_pca_2d_scatter.png', dpi=120, bbox_inches='tight')
    plt.close()
    log("  -> dbscan_02_pca_2d_scatter.png saved")


# Plot 3: Cross-tabulation Heatmap (same style as K-Means version)
def plot_crosstab_heatmap(crosstab):
    crosstab_norm = crosstab.div(crosstab.sum(axis=1), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues',
                cbar_kws={'label': 'Count'}, ax=axes[0])
    axes[0].set_title('Cluster vs Attack Category (Counts)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Attack Category')
    axes[0].set_ylabel('Cluster (-1 = noise)')

    sns.heatmap(crosstab_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                cbar_kws={'label': 'Proportion'}, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title('Cluster Composition (Row-normalized)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Attack Category')
    axes[1].set_ylabel('Cluster (-1 = noise)')

    plt.tight_layout()
    plt.savefig('dbscan_03_crosstab_heatmap.png', dpi=120, bbox_inches='tight')
    plt.close()
    log("  -> dbscan_03_crosstab_heatmap.png saved")


# Main
if __name__ == "__main__":
    # 1. Load data
    X_attack, y_cat_attack = load_attack_data()

    # 2. PCA reduction (same as K-Means version)
    X_pca, pca_model = apply_pca(X_attack)
    X_pca_global = X_pca  # used by evaluate() for silhouette

    # 3. Find eps (k-distance graph + sweep)
    k_dist, sweep_results = find_optimal_eps(X_pca)

    # Pick FINAL_EPS automatically: among candidates, choose the one whose
    # cluster count is closest to 8 (to match the 8 real attack categories),
    # breaking ties by lower noise. If none produce >=2 clusters, keep default.
    valid = [(eps, nc, nn, npct) for (eps, nc, nn, npct) in sweep_results if nc >= 2]
    if valid:
        FINAL_EPS = min(valid, key=lambda r: (abs(r[1] - 8), r[3]))[0]
    log(f"\n  -> Selected FINAL_EPS = {FINAL_EPS}")

    # 4. Final clustering
    labels, db_model = final_clustering(X_pca, FINAL_EPS)

    # 5. Evaluation
    ari_all, nmi_all, ari_clean, nmi_clean, crosstab = evaluate(labels, y_cat_attack)

    # 6. Visualization
    log("\n" + "=" * 60)
    log("[Step 6] Save plots")
    log("=" * 60)
    plot_k_distance(k_dist)
    plot_pca_2d(X_attack.values, labels, y_cat_attack)
    plot_crosstab_heatmap(crosstab)

    # 7. Save log file
    with open('dbscan_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

    log("\n" + "=" * 60)
    log("DBSCAN clustering done")
    log("=" * 60)
    log(f"  eps: {FINAL_EPS} | min_samples: {MIN_SAMPLES}")
    log(f"  ARI (all): {ari_all:.4f} | NMI (all): {nmi_all:.4f}")
    log(f"  Output files: dbscan_01~03*.png, dbscan_report.txt")
