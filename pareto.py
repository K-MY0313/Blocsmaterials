# pareto.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def pareto_front_mask(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """
    cols に挙げた指標を「全部大きいほど良い」とみなして、
    パレートフロントに属する行だけ True になるマスクを返す。
    """
    data = df[cols].values
    n = data.shape[0]
    is_efficient = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_efficient[i]:
            continue

        # i が j を支配する条件：
        #  全ての指標で data[i] >= data[j]
        #  かつ少なくとも1つは > である
        dominates = np.all(data[i] >= data, axis=1) & np.any(
            data[i] > data, axis=1
        )

        # i が支配する点はフロントから除外
        is_efficient[dominates] = False
        # 自分自身は残す
        is_efficient[i] = True

    return is_efficient


def plot_pareto(df: pd.DataFrame, cols: list[str], title: str):
    mask = pareto_front_mask(df, cols)
    front = df[mask]

    x_col, y_col = cols

    print(f"\n=== Pareto front for {cols} ===")
    print(f"件数: {front.shape[0]}")
    print(front[[x_col, y_col, "cluster", "cluster_name"]].sort_values(x_col))

    # 可視化
    plt.figure()
    plt.scatter(
        df[x_col],
        df[y_col],
        alpha=0.3,
        label="others",
    )
    plt.scatter(
        front[x_col],
        front[y_col],
        label="Pareto front",
        edgecolors="k",
    )
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv("materials_with_cluster.csv")

    # 1) 強度 vs 加工性 のフロント
    plot_pareto(
        df,
        ["strength", "workability"],
        "Pareto front: strength vs workability",
    )

    # 2) 強度 vs 耐熱 のフロント（両方「高いほど良い」とみなす）
    plot_pareto(
        df,
        ["strength", "heat"],
        "Pareto front: strength vs heat",
    )


if __name__ == "__main__":
    main()
