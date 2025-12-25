# bestiary_v0_2.py
import argparse
import json
import math
import random
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd

from generator import random_elements
from material import Material


Recipe = Tuple[int, int, int]


PROFILES_PATH = "profiles.json"


def load_profiles(path: str = PROFILES_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_feature(summary: dict, feature: str) -> float:
    # 例: quantum_drift_abs -> abs(quantum_drift)
    if feature.endswith("_abs"):
        base = feature[:-4]
        return abs(summary[base])
    return summary[feature]


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
    weights = profile.get("weights", {})
    filters = profile.get("filters", {})

    if not passes_filters(summary, filters):
        return float("-inf")

    score = 0.0
    for feat, w in weights.items():
        score += w * _get_feature(summary, feat)
    return score


def aggregate_summary(summaries: List[dict]) -> dict:
    """複数回 summary を平均化して1つにまとめる（表示用）"""
    if not summaries:
        return {}
    keys = summaries[0].keys()
    out = {}
    for k in keys:
        out[k] = float(np.mean([s[k] for s in summaries]))
    return out


def random_recipe(num_elems: int) -> Recipe:
    return (
        random.randrange(num_elems),
        random.randrange(num_elems),
        random.randrange(num_elems),
    )


def crossover(r1: Recipe, r2: Recipe) -> Tuple[Recipe, Recipe]:
    cut = random.randint(1, 2)
    c1 = r1[:cut] + r2[cut:]
    c2 = r2[:cut] + r1[cut:]
    return c1, c2


def mutate(r: Recipe, num_elems: int, p: float) -> Recipe:
    lst = list(r)
    for i in range(len(lst)):
        if random.random() < p:
            lst[i] = random.randrange(num_elems)
    return tuple(lst)  # type: ignore[return-value]


def eval_recipe(recipe: Recipe, elems: List, profile: dict, repeats: int = 5) -> Tuple[float, dict]:
    i, j, k = recipe

    sums = []
    for _ in range(repeats):
        mat = Material(elems[i], elems[j], elems[k])
        sums.append(mat.summary())

    agg = aggregate_summary(sums)           # 平均 summary
    score = evaluate_with_profile(agg, profile)  # filtersも平均に対して判定される

    return score, agg



def run_ga_for_profile(
    elems: List,
    profile: dict,
    repeats: int,
    pop_size: int,
    generations: int,
    elite_size: int,
    mut_rate: float,
) -> Dict[str, Any]:

    num_elems = len(elems)
    population: List[Recipe] = [random_recipe(num_elems) for _ in range(pop_size)]

    best_recipe = None
    best_score = float("-inf")
    best_summary = None

    for _gen in range(generations):
        scored = []
        for r in population:
            score, summary = eval_recipe(r, elems, profile, repeats=repeats)
            scored.append((score, r, summary))

            if score > best_score:
                best_score = score
                best_recipe = r
                best_summary = summary

        scored.sort(key=lambda x: x[0], reverse=True)

        elites = scored[:elite_size]
        elite_recipes = [r for _, r, _ in elites]

        new_pop: List[Recipe] = elite_recipes.copy()
        while len(new_pop) < pop_size:
            p1 = random.choice(elite_recipes)
            p2 = random.choice(elite_recipes)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1, num_elems, mut_rate)
            c2 = mutate(c2, num_elems, mut_rate)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop

    return {
        "best_recipe": best_recipe,
        "best_score": best_score,
        "best_summary": best_summary,
    }


def recipe_names(elems: List, recipe: Recipe) -> List[str]:
    i, j, k = recipe
    return [elems[i].name, elems[j].name, elems[k].name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="struct,high_temp,outer,catalyst,superconductor",
                    help="カンマ区切り。例: struct,high_temp,outer")
    ap.add_argument("--num-elems", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=5)

    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--elite", type=int, default=10)
    ap.add_argument("--mut", type=float, default=0.2)

    ap.add_argument("--out-csv", default="bestiary_v0_2.csv")
    ap.add_argument("--out-md", default="bestiary_v0_2.md")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    profiles = load_profiles()
    keys = [p.strip() for p in args.profiles.split(",") if p.strip()]
    for k in keys:
        if k not in profiles:
            raise ValueError(f"Unknown profile '{k}'. choices={list(profiles.keys())}")

    # 同一世界（同一 Element プール）で比較する
    elems = random_elements(args.num_elems)

    rows = []
    for key in keys:
        prof = profiles[key]
        label = prof.get("label", key)

        res = run_ga_for_profile(
            elems=elems,
            profile=prof,
            repeats=args.repeats,
            pop_size=args.pop,
            generations=args.gens,
            elite_size=args.elite,
            mut_rate=args.mut,
        )

        best_r = res["best_recipe"]
        if best_r is None:
            best_names = ["N/A", "N/A", "N/A"]
            summ = res["best_summary"] or {}
            recipe_idx_str = "N/A"
        else:
            best_names = recipe_names(elems, best_r)
            summ = res["best_summary"] or {}
            recipe_idx_str = str(best_r)

        rows.append({
            "profile": key,
            "label": label,
            "recipe_indices": str(best_r),
            "recipe_names": "-".join(best_names),
            "score": res["best_score"],
            "strength": summ.get("strength", math.nan),
            "heat": summ.get("heat", math.nan),
            "stability": summ.get("stability", math.nan),
            "workability": summ.get("workability", math.nan),
            "corrosion": summ.get("corrosion", math.nan),
            "quantum_drift": summ.get("quantum_drift", math.nan),
        })

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    # Markdown も生成
    md_lines = []
    md_lines.append("# Brocks 素材図鑑 v0.2（用途別 最適レシピ）\n")
    md_lines.append(f"- Element pool: num_elems={args.num_elems}, seed={args.seed}\n")
    md_lines.append(f"- GA: pop={args.pop}, gens={args.gens}, elite={args.elite}, mut={args.mut}, repeats={args.repeats}\n\n")
    md_lines.append(df.to_markdown(index=False))
    md_lines.append("\n")

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"[OK] Saved: {args.out_csv}")
    print(f"[OK] Saved: {args.out_md}")
    print(df)


if __name__ == "__main__":
    main()
