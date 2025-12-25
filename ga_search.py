# ga_search.py
import random
from typing import Tuple, List

from generator import random_elements
from material import Material
import json
import argparse
import math
import numpy as np

PROFILES_PATH = "profiles.json"

def load_profiles(path: str = PROFILES_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# === 評価関数（この世界の“総合スコア”） ===
def _get_feature(summary: dict, feature: str) -> float:
    # 例: quantum_drift_abs -> abs(quantum_drift)
    if feature.endswith("_abs"):
        base = feature[:-4]
        return abs(summary[base])
    return summary[feature]

def aggregate_summary(summaries: list[dict]) -> dict:
    """複数回の summary を平均化して1つにまとめる（表示＆記録用）"""
    if not summaries:
        return {}

    keys = summaries[0].keys()
    out = {}
    for k in keys:
        vals = [s[k] for s in summaries]
        # 数値しか入ってない前提（あなたの summary は数値のみ）
        out[k] = float(np.mean(vals))
    return out
def passes_filters(summary: dict, filters: dict | None) -> bool:
    if not filters:
        return True

    for k, th in filters.items():
        if k.endswith("_min"):
            feat = k[:-4]
            if _get_feature(summary, feat) < th:
                return False
        elif k.endswith("_max"):
            feat = k[:-4]
            if _get_feature(summary, feat) > th:
                return False
        else:
            # 拡張用：今は無視
            pass
    return True


def evaluate_with_profile(summary: dict, profile: dict) -> float:
    """
    profiles.json の1プロファイルを使ってスコアを計算。
    filters に落ちたら -inf にして GA が選ばないようにする。
    """
    weights = profile.get("weights", {})
    filters = profile.get("filters", {})

    if not passes_filters(summary, filters):
        return float("-inf")

    score = 0.0
    for feat, w in weights.items():
        score += w * _get_feature(summary, feat)
    return score

# === レシピ表現 ===材料を作るための“設計図（
Recipe = Tuple[int, int, int]  # Recipeは elements のインデックス3つ（重複可）

#ランダムに元素を3つ選ぶ
def random_recipe(num_elems: int) -> Recipe:
    # 同じ元素を複数回使うのも許可
    return (
        random.randrange(num_elems),
        random.randrange(num_elems),
        random.randrange(num_elems),
    )

#保管されたレシピ(material)の組み合わせ
def crossover(r1: Recipe, r2: Recipe) -> Tuple[Recipe, Recipe]:
    # 1点交叉（位置1 or 2で切り替え）
    cut = random.randint(1, 2)
    c1 = r1[:cut] + r2[cut:]
    c2 = r2[:cut] + r1[cut:]
    return c1, c2

# 突然変異（各遺伝子座を確率pで別の要素に置換）
def mutate(r: Recipe, num_elems: int, p: float) -> Recipe:
    lst = list(r)
    for i in range(len(lst)):
        if random.random() < p:
            lst[i] = random.randrange(num_elems)
    return tuple(lst)  # type: ignore[return-value]

# 遺伝子型(Recipe) → 表現型(Material) へ変換
def recipe_to_material(recipe: Recipe, elems: List) -> Material:
    i, j, k = recipe
    return Material(elems[i], elems[j], elems[k])
def aggregate_summary(summaries: list[dict]) -> dict:
    """複数回の summary を平均化して1つにまとめる（表示用）"""
    if not summaries:
        return {}

    keys = summaries[0].keys()
    out = {}
    for k in keys:
        out[k] = float(np.mean([s[k] for s in summaries]))
    return out

#出した素材のパラメータを評価関数に入れて評価する
def eval_recipe(recipe: Recipe, elems: list, profile: dict, repeats: int = 5) -> tuple[float, dict]:
    scores = []
    sums = []

    for _ in range(repeats):
        i, j, k = recipe
        mat = Material(elems[i], elems[j], elems[k])
        summary = mat.summary()

        s = evaluate_with_profile(summary, profile)

        # filters落ち（-inf）は即失格扱いにする
        if s == float("-inf"):
            return float("-inf"), summary

        scores.append(s)
        sums.append(summary)

    return float(np.mean(scores)), aggregate_summary(sums)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="overall", help="profiles.json のキー名")
    args = parser.parse_args()

    profiles = load_profiles()
    if args.profile not in profiles:
        raise ValueError(f"Unknown profile: {args.profile}. choices={list(profiles.keys())}")

    active_profile = profiles[args.profile]
    print(f"[INFO] Using profile: {args.profile} ({active_profile.get('label','')})")
    #ランダムな値にシード値を入れる
    random.seed(42)

    # === 元になる Element プールを作る ===
    num_elems = 20
    elems = random_elements(num_elems)

    # === GA パラメータ ===
    POP_SIZE = 50
    GENERATIONS = 40
    ELITE_SIZE = 10
    MUT_RATE = 0.2

    # 初期集団
    population: List[Recipe] = [
        random_recipe(num_elems) for _ in range(POP_SIZE)
    ]

    best_overall = None
    best_overall_score = float("-inf")
    best_overall_summary = None

    for gen in range(GENERATIONS):
        # 評価
        scored = []
        for r in population:
            score, summary = eval_recipe(r, elems, active_profile ,repeats=5)
            scored.append((score, r, summary))

            if score > best_overall_score:
                best_overall_score = score
                best_overall = r
                best_overall_summary = summary

        scored.sort(key=lambda x: x[0], reverse=True)

        print(
            f"Gen {gen:02d}: "
            f"best_score={scored[0][0]:.2f}, "
            f"recipe={scored[0][1]}"
        )

        # エリート選択
        elites = scored[:ELITE_SIZE]
        elite_recipes = [r for _, r, _ in elites]

        # 新しい集団を作る
        new_pop: List[Recipe] = elite_recipes.copy()

        while len(new_pop) < POP_SIZE:
            #エリートをランダムで選ぶ
            p1 = random.choice(elite_recipes)
            p2 = random.choice(elite_recipes)
            #ランダムで選んだエリートmaterialを交叉させる
            c1, c2 = crossover(p1, p2)
            #確率で突然変異
            c1 = mutate(c1, num_elems, MUT_RATE)
            c2 = mutate(c2, num_elems, MUT_RATE)
            #c1を格納
            new_pop.append(c1)
            #空きがあればc2を格納
            if len(new_pop) < POP_SIZE:
                new_pop.append(c2)
        #新しいく生まれた遺伝子を入れる
        population = new_pop

    # === 結果表示 ===
    print("\n=== Best recipe over all generations ===")
    print(f"recipe indices: {best_overall}")
    if best_overall is not None:
        i, j, k = best_overall
        e_names = [elems[i].name, elems[j].name, elems[k].name]
        print(f"Element names: {e_names}")
    print(f"score: {best_overall_score:.2f}")
    print("summary:", best_overall_summary)


if __name__ == "__main__":
    main()
