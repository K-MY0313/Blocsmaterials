import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# === データ読み込み ===
df = pd.read_csv("materials.csv")

# === クラスタリング ===
k = 4  # クラスター数（世界観に合わせて自由に変えてOK）
kmeans = KMeans(n_clusters=k, random_state=0)
df["cluster"] = kmeans.fit_predict(df)
# === 特徴量一覧 ===
features = ["strength", "heat", "stability", "workability", "corrosion", "quantum_drift"]

# クラスタごとの平均値（系統サマリ）
cluster_stats = df.groupby("cluster")[features].mean()
print("=== Cluster stats ===")
print(cluster_stats)
print(df.head())  # 確認用

# === プロット ===
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="strength",
    y="stability",
    hue="cluster",
    palette="tab10",
    s=60
)
plt.title("Strength vs Stability (Clustered)")
plt.legend(title="Cluster")
plt.show()
plt.figure()
scatter = plt.scatter(
    df["heat"],
    df["stability"],
    c=df["cluster"],
    cmap="tab10"
)
plt.xlabel("heat")
plt.ylabel("stability")
plt.title("Heat vs Stability (Clustered)")
plt.colorbar(scatter, label="Cluster")
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

p = ax.scatter(
    df["strength"],
    df["heat"],
    df["stability"],
    c=df["cluster"],
)

ax.set_xlabel("strength")
ax.set_ylabel("heat")
ax.set_zlabel("stability")
ax.set_title("Material Landscape (3D)")

fig.colorbar(p, label="Cluster")
plt.show()