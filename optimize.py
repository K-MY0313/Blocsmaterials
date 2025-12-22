# optimize.py
import pandas as pd
import json
import math

PROFILES_PATH = "profiles.json"

# === 1. データ読み込み ===
df = pd.read_csv("materials_with_cluster.csv")

# === 2. プロファイル定義を読み込み ===
with open(PROFILES_PATH, "r", encoding="utf-8") as f:
    profiles = json.load(f)


# === 3. 汎用的なスコア計算関数 ===

def row_passes_filters(row, filters: dict) -> bool:
    """filters に従ってこの行を採用するかどうかを判定"""
    if not filters:
        return True

    # 実数値を取り出すヘルパ
    def get_value(col_name: str) -> float:
        if col_name == "quantum_drift_abs":
            return abs(row["quantum_drift"])
        else:
            return row[col_name]

    for key, threshold in filters.items():
        if key.endswith("_min"):
            col = key[:-4]  # "_min" を取る
            if get_value(col) < threshold:
                return False
        elif key.endswith("_max"):
            col = key[:-4]
            if get_value(col) > threshold:
                return False
        else:
            # それ以外はとりあえず無視（拡張用）
            pass

    return True


def calc_score(row, weights: dict) -> float:
    """weights に従ってスコアを計算（単純な線形結合）"""
    score = 0.0

    for feature, w in weights.items():
        if feature == "quantum_drift_abs":
            v = abs(row["quantum_drift"])
        else:
            v = row[feature]
        score += w * v

    return score


# === 4. 各プロファイルについてスコア列を追加 ===

for name, cfg in profiles.items():
    weights = cfg.get("weights", {})
    filters = cfg.get("filters", {})

    score_col = f"score_{name}"

    # 一旦 NaN を入れて、条件を満たす行だけ計算する
    df[score_col] = math.nan

    mask = df.apply(lambda r: row_passes_filters(r, filters), axis=1)
    df.loc[mask, score_col] = df[mask].apply(
        lambda r: calc_score(r, weights), axis=1
    )

    # ついでにランキング上位だけ表示
    top_n = 10
    print(f"\n=== Profile: {name} ({cfg.get('label', '')}) / top {top_n} ===")
    cols_show = [
        score_col, "cluster", "cluster_name",
        "strength", "heat", "stability",
        "workability", "corrosion", "quantum_drift"
    ]
    top = df[mask].sort_values(score_col, ascending=False).head(top_n)
    print(top[cols_show])


# === 5. 保存 ===
df.to_csv("materials_ranked_profiles.csv", index=False)
print("\nSaved ranked profiles to materials_ranked_profiles.csv")
