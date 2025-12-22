import random
import pandas as pd
from element import Element
from material import Material

def random_elements(n=12):
    names = ["A", "B", "C", "D"]  # 個性定義済み
    # 12 個欲しいときは同じ名前を複数作って OK
    return [Element(random.choice(names)) for _ in range(n)]

def random_material(elements):
    e1, e2, e3 = random.sample(elements, 3)
    return Material(e1, e2, e3)

if __name__ == "__main__":
    elems = random_elements(20)
    mats = [random_material(elems) for _ in range(200)]

    # DataFrame へ
    df = pd.DataFrame([m.summary() for m in mats])
    print(df.head())
    print(df.describe())
    df[["strength", "workability"]].corr()
    # 保存したい場合
    df.to_csv("materials.csv", index=False)


