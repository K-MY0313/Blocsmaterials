# map.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# === 1. データ読み込み ===
df = pd.read_csv("materials.csv")

# 特徴量カラム
features = ["strength", "heat", "stability",
            "workability", "corrosion", "quantum_drift"]
X = df[features].values

# === 2. クラスタリング ===
k = 4  # plot.py と揃えとく
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(X)
df["cluster"] = labels

# === 3. クラスタごとの平均値（系統サマリ） ===
cluster_stats = df.groupby("cluster")[features].mean()
print("=== Cluster stats ===")
print(cluster_stats)

# === 4. 代表マテリアル（クラスタ中心に最も近いもの） ===
centers = kmeans.cluster_centers_

rep_indices = []
for c in range(k):
    idx = np.where(df["cluster"].values == c)[0]
    dist = np.linalg.norm(X[idx] - centers[c], axis=1)
    rep_indices.append(idx[dist.argmin()])

rep_materials = df.iloc[rep_indices]
print("\n=== Representative materials ===")
print(rep_materials[features + ["cluster"]])

# === 5. （おまけ）クラスタ名の枠だけ用意しておく ===
cluster_name_map = {
    0: "Cluster0",
    1: "Cluster1",
    2: "Cluster2",
    3: "Cluster3",
}
df["cluster_name"] = df["cluster"].map(cluster_name_map)

# 必要なら保存
df.to_csv("materials_with_cluster.csv", index=False)
print("\nSaved to materials_with_cluster.csv")
print("\n=== Brocks 素材図鑑 v0.1 ===")
for c in range(k):
    name = cluster_name_map[c]
    stats = cluster_stats.loc[c]
    rep = rep_materials[rep_materials["cluster"] == c].iloc[0]

    print(f"\n[{c}] {name}")
    print(f"  平均値: "
          f"Str={stats['strength']:.1f}, "
          f"Heat={stats['heat']:.1f}, "
          f"Stab={stats['stability']:.1f}, "
          f"Work={stats['workability']:.1f}, "
          f"Corr={stats['corrosion']:.1f}, "
          f"Q={stats['quantum_drift']:.2f}")
    print("  代表マテリアル例:")
    print(f"    index={rep.name}, "
          f"Str={rep['strength']:.1f}, Heat={rep['heat']:.1f}, "
          f"Stab={rep['stability']:.1f}, Work={rep['workability']:.1f}")
